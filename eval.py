# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:04:48 2020

@author: 이상헌
"""

from __future__ import unicode_literals, print_function, division
import random

from utils import tensorFromSentence
import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 100
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, dictionary, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(dictionary, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(dictionary.index2token[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
    
    
def evaluateRandomly(encoder, decoder, pairs, dictionary, fp, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        fp.write('> ' + pair[0] + '\n')
        print('=', pair[1])
        fp.write('>= ' + pair[1] + '\n')
        output_words, attentions = evaluate(encoder, decoder, pair[0], dictionary)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        fp.write('< ' + output_sentence + '\n\n')
        print('')
