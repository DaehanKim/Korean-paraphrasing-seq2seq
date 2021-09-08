
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

MAX_LENGTH = 100

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedtable):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedtable = nn.Embedding(*embedtable.shape)
        self.embedtable.weight.data.copy_(embedtable)
        self.gru = nn.GRU(hidden_size, hidden_size)
        

    def forward(self, input, hidden):
        # embedding함수 --> lookup table에서 찾는 걸로 바꾸기
        
        embedded = self.embedtable(input) # view=reshape
        #print(input.device(), self.embedtable.weight.device())
        output = torch.unsqueeze(embedded, 0)
        output = output.to(DEVICE)
        #print(output.size(), hidden.size())
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(DEVICE)
        
    
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedtable, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.word_dim = embedtable.size(1)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedtable = nn.Embedding(*embedtable.shape)
        self.embedtable.weight.data.copy_(embedtable)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        ## copy mechanism
        self.do_gen = nn.Sequential(nn.Linear(hidden_size*2 + self.word_dim, 1), nn.Sigmoid()) # do-generation probability
        self.copy_weight_proj = nn.Linear(hidden_size*2, 1)

        ## copy mechanism
    def get_alpha(self, e_tokens, e_states, d_state):
        input_ = torch.cat([e_states, d_state.squeeze(0).repeat(self.max_length,1)], dim=1)
        alphas = F.softmax(self.copy_weight_proj(input_)[:e_tokens.size(0),:], dim=0)
        return alphas

    def get_context(self, e_tokens, e_states, alphas):
        context = (e_states[:e_tokens.size(0),:]*alphas).sum(dim=0)
        return context



    def forward(self, input, hidden, e_tokens, encoder_outputs):

        if input.dim() == 1 : 
            input = input.unsqueeze(0)
        elif input.dim() == 0:
            input = torch.tensor([[input]]).long().to(DEVICE)

        embedded = self.embedtable(input)

        embedded = self.dropout(embedded)

        embedded = embedded.to(DEVICE)
            
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output_prob = F.softmax(self.out(output[0]), dim=1) # generation 
        # print(output_prob.size())
        # probabilities for each tokens
        alphas = self.get_alpha(e_tokens,encoder_outputs, output) 
        context = self.get_context(e_tokens,encoder_outputs, alphas)
        # print(alphas)
        # print(e_tokens.size(), alphas.size())
        copy_prob = torch.zeros(1,output_prob.size(1)).to(DEVICE).scatter_add(1, e_tokens.permute([1,0]), alphas.permute([1,0]))
        # print(copy_prob)

        # print(output.size(), encoder_outputs.size(), context.size(), embedded.size())
        # exit()
        mix_ratio = self.do_gen(torch.cat([output[-1,-1,:], context, embedded.squeeze(0).squeeze(0)],dim=0))


        prob = output_prob*mix_ratio 
        # print(prob.size())
        # exit()
        prob[0,e_tokens[:,0]] += alphas.squeeze(1)*(1-mix_ratio)


        return prob, hidden, attn_weights


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(DEVICE)
