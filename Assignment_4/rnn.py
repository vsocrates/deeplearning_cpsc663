
from __future__ import unicode_literals, print_function, division

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import re

from io import open
import glob
import os
import unicodedata
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#############################################
# TODO 3: define an lstm encoder function that takes the embedding lookup and produces a final state
# _, final_state = your_lstm_encoder(embedding_lookup_for_x, ...)
#############################################
class LSTMEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.embed_dim = embed_dim 
        self.hidden_dim = hidden_dim
         
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return (h_n, c_n)




#############################################
# TODO 4: define an lstm decoder function that takes the final state from previous step and produces a sequence of outputs
# outs, _ = your_lstm_decoder(final_state, ...)
#############################################
class LSTMDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_size):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
        self.output_size = output_size 
        self.hidden_dim = hidden_dim 
        
    def forward(self, x, hidden, cell):
#         print("first x", x.shape)
#         x = x.reshape(1, x.shape[0], x.shape[1])
#         print("2nd x", x.shape)      
#         print('hideen', hidden.shape)
#         print("cedll", cell.shape)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
#         print("output decoder", output.shape)
#         print("output decoder", output[0].shape)        
        output = self.out(output.squeeze(0))
#         print("output decoder", output.shape)        
#         output = self.softmax(output)
#         print("output decoder", output.shape)        
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, embedding):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding

        assert encoder.hidden_dim == decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal"
        
    def forward(self, x):
        
        batch_size = x.shape[1]
        input_len = x.shape[0]        
        vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(input_len, batch_size, vocab_size)
        
        embedded = self.embedding(x)
#         print("embed", embedded.shape)
        hidden, cell = self.encoder(embedded)
#         print("output", output.shape)
#         print("output", hidden.shape)
#         print("output", cell.shape)        
        # first input is 0 vector as SOS token 
        input_val = torch.zeros(1, batch_size, dtype=torch.int64)
#         print("input1", input_val.shape)
        # we start the decoder at 1, so the first row is all 0s (SOS token)
        for t in range(0, input_len):
            input_val = self.embedding(input_val)
#             print("input", input_val.shape)
            output, hidden, cell = self.decoder(input_val, hidden, cell)            
            outputs[t] = output    
#             max_vals, max_idxs = torch.max(output, 1)
            max_idxs = output.argmax(1)
#             print("output shape?1", output.shape)
#             print("output shape?2", output)
            
            input_val = max_idxs
            input_val = input_val.unsqueeze(0)
#             print("input val", input_val.shape)
#             print("post input val", input_val)
#             print("TTTT", t)
        return outputs

# helper function
def to_one_hot(y_tensor, c_dims):
    """converts a N-dimensional input to a NxC dimnensional one-hot encoding
    """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    c_dims = c_dims if c_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], c_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y_tensor.shape, -1)
    return y_one_hot
