import os
import pickle, dill

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.helpers import (get_character_encoder,
                         get_character_decoder)


class BigramModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, dropout=0.2):
        super(BigramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.nn_linear = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, targets=None):
        try:
            X = self.embedding(inputs)  # output shape: (B, T, emb_dim)
        except:
            print(inputs)
            raise "Issue in embedding layer"
        X = F.relu(self.nn_linear(X))  # output shape: (B, T, vocab_size)
        logits = self.dropout(X)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens=128):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(inputs)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            inputs = torch.cat((inputs, index_next), dim=1)  # (B, T+1)
        return inputs


# EncoderDecoderModule.py
def get_bigram_encoder_decoder(charset):
    char_encoder = get_character_encoder(charset)
    char_decoder = get_character_decoder(char_encoder)

    def encode(s):
        return [char_encoder[c] for c in s]

    def decode(l):
        return ''.join([char_decoder[i] for i in l])

    return encode, decode


def load_encoder_decoder_functions():
    if os.path.exists(os.path.abspath('models/bigram_encoder_decoder.pkl')):
        with open(os.path.abspath('models/bigram_encoder_decoder.pkl'), 'rb') as file:
            encode, decode = dill.load(file)
            return encode, decode
    else:
        raise FileNotFoundError("Encoder and decoder functions file not found.")
