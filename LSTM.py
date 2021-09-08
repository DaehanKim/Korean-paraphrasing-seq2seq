
from __future__ import unicode_literals, print_function, division

import torch
import numpy as np

from models import Encoder, AttnDecoder
from train import trainIters
from eval import evaluateRandomly
from utils import *
from config import *



file_name = "new_data.xlsx"

dictionary, pair_data = prepareData("kor", file_name)
embedtable = np.loadtxt("word_emb.txt", delimiter=" ", dtype='float32')
special_embeddings = np.concatenate((np.random.rand(len(SPECIAL_TOKENS)-1, 128).astype('float32'),
	np.zeros((1,128), dtype=np.float32)), axis=0)
embedtable = np.insert(embedtable, [2], special_embeddings, axis=0)
print("index for OOV token : ", embedtable.shape[0])
embedtable = np.concatenate((embedtable, np.random.rand(1, 128).astype('float32')), axis=0) # embedding for OOV token
embedtable = torch.from_numpy(embedtable).float()


encoder = Encoder(dictionary.n_tokens, 128, embedtable).to(DEVICE)
attndecoder = AttnDecoder(128, dictionary.n_tokens, embedtable, dropout_p=0.1).to(DEVICE)

trainIters(encoder, attndecoder, dictionary, pair_data, epochs=1000)
evaluateRandomly(encoder, attndecoder, pair_data, dictionary, n=10)
