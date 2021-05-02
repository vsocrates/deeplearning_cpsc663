
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



BATCH_SIZE = 10
ITERATIONS = 1000
SEQ_LENGTH = 50
EMBEDDING_SIZE = 100
LSTM_SIZE = 64


# TODO 1: put a text file of your choosing in the same directory and put its name here
# TEXT_FILE = 'yourfile.txt'

string = open(TEXT_FILE).read()

# convert text into tekens
tokens = re.split('\W+', string)

# get vocabulary
vocabulary = sorted(set(tokens))

# get corresponding indx for each word in vocab
word_to_ix = {word: i for i, word in enumerate(vocabulary)}

VOCABULARY_SIZE = len(vocabulary)
print('vocab size: {}'.format(VOCABULARY_SIZE))


#############################################
# TODO 2: create variable for embedding matrix. Hint: you can use nn.Embedding for this
#############################################
embedding = nn.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE)

#############################################
# TODO 3: define an lstm encoder function that takes the embedding lookup and produces a final state
# _, final_state = your_lstm_encoder(embedding_lookup_for_x, ...)
#############################################
class LSTMEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.embed_dim = embed_dim 
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)




#############################################
# TODO 4: define an lstm decoder function that takes the final state from previous step and produces a sequence of outputs
# outs, _ = your_lstm_decoder(final_state, ...)
#############################################
class LSTMDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_size):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.output_size = output_size 
        
    def forward(self, x, hidden, cell):
        output, hidden = self.lstm(x, (hidden, cell))
        output = self.softmax(self.out(output[0]))
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, embedding):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        
    def forward(self, x):
        
        batch_size = x.shape[1]
        input_len = x.shape[0]        
        vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(input_len, batch_size, vocab_size)
        
        embedded = self.embedding(x)
        output, (hidden, cell) = self.encoder(embedded)
        
        # first input is 0 vector as SOS token 
        input_val = torch.zeros(1, self.encoder.embed_dim)
        
        # we start the decoder at 1, so the first row is all 0s (SOS token)
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(self.embedding(input_val), hidden, cell)            
            
            outputs[t] = output

        return outputs

#############################################
# TODO: create loss/train ops
#############################################

# ex.
encoder = LSTMEncoder(EMBEDDING_SIZE, LSTM_SIZE)
decoder = LSTMDecoder(embed_dim, hidden_dim, VOCABULARY_SIZE)
model = Seq2Seq(encoder, decoder, embedding)
loss_fn = nn.BCELoss()
# loss = loss_fn(out,one_hots)
optimizer = optim.Adam(model.parameters(),lr=0.01)
    

# helper function
def to_one_hot(y_tensor, c_dims):
    """converts a N-dimensional input to a NxC dimnensional one-hot encoding
    """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    c_dims = c_dims if c_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], c_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y_tensor.shape, -1)
    return y_one_hot


# do training

i = 0
for num_iter in range(ITERATIONS):

    if num_iter % 10 == 0: print(num_iter)
    batch = [[vocabulary.index(v) for v in tokens[ii:ii + SEQ_LENGTH]] for ii in range(i, i + BATCH_SIZE)]
    batch = np.stack(batch, axis=0)
    batch = torch.tensor(batch, dtype=torch.long)
    i += BATCH_SIZE
    if i + BATCH_SIZE + SEQ_LENGTH > len(tokens): i = 0

    #############################################
    #TODO: create loss and update step
    #############################################
    optimizer.zero_grad()
    
    outputs = model(batch)
    print(outputs)
    loss = loss_fn(out,to_one_hot(outputs))
    optimizer.step()
    
    
    # Hint:following steps will most likely follow the pattern
    # out = model(batch)
    # loss = loss_function(out,batch)
    # optimizer.step()
    # optimizer.zero_grad()

    
# plot word embeddings
# assuming embeddings called "learned_embeddings",

fig = plt.figure()
learned_embeddings_pca = sklearn.decomposition.PCA(2).fit_transform(learned_embeddings)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(learned_embeddings_pca[:, 0], learned_embeddings_pca[:, 1], s=5, c='w')
MIN_SEPARATION = .1 * min(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])

fig.clf()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(learned_embeddings_pca[:, 0], learned_embeddings_pca[:, 1], s=5, c='w')


#############################################
# TODO 5: run this multiple times
#############################################

xy_plotted = set()
for i in np.random.choice(VOCABULARY_SIZE, VOCABULARY_SIZE, replace=False):
    x_, y_ = learned_embeddings_pca[i]
    if any([(x_ - point[0])**2 + (y_ - point[1])**2 < MIN_SEPARATION for point in xy_plotted]): continue
    xy_plotted.add(tuple([learned_embeddings_pca[i, 0], learned_embeddings_pca[i, 1]]))
    ax.annotate(vocabulary[i], xy=learned_embeddings_pca[i])
