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
        self.nn_linear1 = nn.Linear(emb_dim, vocab_size)
        self.nn_linear2 = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, targets=None):
        try:
            X = self.embedding(inputs)  # output shape: (B, T, emb_dim)
        except:
            print(inputs)
            raise "Issue in embedding layer"
        X = self.dropout(X)  # output shape: (B, T, emb_dim)
        X = F.relu(self.nn_linear1(X))  # output shape: (B, T, vocab_size)
        logits = self.nn_linear2(X)  # output shape: (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # becomes (B*T, C)
            targets = targets.view(B * T)  # becomes (B*T)
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
