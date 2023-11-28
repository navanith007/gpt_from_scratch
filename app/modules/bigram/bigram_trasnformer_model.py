import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is the implementation of the transformer model for the GPT model
The model can be explained as follows:
1. Embedding layer: This layer takes in the input and returns the embedding
2. Positional encoding: This layer takes in the embedding and returns the positional encoding
3. Transformer block: This layer takes in the positional encoding and returns the output of the transformer block
4. Linear layer: This layer takes in the output of the transformer block and returns the logits
5. Loss: This layer takes in the logits and the targets and returns the loss
6. Generate: This layer takes in the input and returns the generated output
7. GPT: This is the main model that combines all the above layers and returns the output
"""


class EmbeddingLayer(nn.Module):
    """
    Embedding layer for the transformer model
    This is a simple embedding layer that takes in the input and returns the embedding
    """

    def __init__(self, vocab_size, emb_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, inputs):
        return self.embedding(inputs)  # (B, T, emb_dim)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    This is a fixed positional encoding based on the formula given in the paper and the formula is as follows:
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    where,
     pos is the position
     i is the dimension
     d_model is the embedding dimension
    """

    def __init__(self, emb_dim, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim  # embedding dimension
        pe = torch.zeros(max_seq_len, emb_dim)  # positional encoding
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))  # (emb_dim/2)
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_seq_len, emb_dim/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_seq_len, emb_dim/2)
        self.register_buffer('pe', pe)  # (max_seq_len, emb_dim)

    def forward(self, inputs):
        inputs = inputs * torch.sqrt(torch.tensor(self.emb_dim, dtype=torch.float))  # (B, T, emb_dim)
        inputs = inputs + self.pe[:inputs.size(1), :]  # (B, T, emb_dim)
        return inputs


class AttentionLayer(nn.Module):
    """
    Attention layer for the transformer model
    This is a simple attention layer that takes in the input and returns the attention
    The mathematical formula for the attention is as follows:
    Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    where,
     Q is the query
     K is the key
     V is the value
     d_k is the dimension of the key
    """

    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv_linear = nn.Linear(emb_dim, emb_dim * 3)
        self.out_linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, inputs, mask=None):
        B, T, C = inputs.shape
        qkv = self.qkv_linear(inputs)  # (B, T, emb_dim*3)
        qkv = qkv.reshape(B, T, 3, self.num_heads, C // self.num_heads)  # (B, T, 3, num_heads, emb_dim//num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, emb_dim//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, T, emb_dim//num_heads)
        dot_prod = torch.einsum('bhid,bhjd->bhij', q, k)  # (B, num_heads, T, T)
        dot_prod = dot_prod / torch.sqrt(torch.tensor(self.emb_dim, dtype=torch.float))  # (B, num_heads, T, T)
        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == 0, float('-inf'))  # (B, num_heads, T, T)
        attention = F.softmax(dot_prod, dim=-1)  # (B, num_heads, T, T)
        attention = self.dropout(attention)  # (B, num_heads, T, T)
        out = torch.einsum('bhij,bhjd->bhid', attention, v)  # (B, num_heads, T, emb_dim//num_heads)
        out = out.permute(1, 2, 0, 3).reshape(B, T, C)  # (B, T, emb_dim)
        out = self.out_linear(out)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer block for the transformer model
    This is a simple transformer block that takes in the input and returns the output of the transformer block
    The mathematical formula for the transformer block is as follows:
    X = Dropout(inputs + Attention(LayerNorm(inputs)))
    Y = Dropout(Feedforward(Activation(FeedForward(LayerNorm(X)))))
    :return: X + Y
    """

    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mha = AttentionLayer(emb_dim, num_heads, dropout)
        self.nn_linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, inputs, mask=None):
        X = self.ln1(inputs)  # layer normalization #shape of inputs is (B, T, emb_dim)
        X = self.mha(X, mask)  # masked self attention #shape of inputs is (B, T, emb_dim)
        X = X + inputs  # residual connection #shape of inputs is (B, T, emb_dim)
        X = self.dropout(X)  # dropout  # shape of inputs is (B, T, emb_dim)
        Y = self.ln2(X)  # layer normalization # shape of inputs is (B, T, emb_dim)
        Y = self.nn_linear(Y)  # feed forward # shape of inputs is (B, T, emb_dim)
        Y = F.gelu(Y)  # gelu activation # shape of inputs is (B, T, emb_dim)
        Y = self.nn_linear(Y)  # feed forward # shape of inputs is (B, T, emb_dim)
        Y = self.dropout(Y)  # dropout # shape of inputs is (B, T, emb_dim)
        return X + Y  # residual connection # shape of inputs is (B, T, emb_dim)


class GPT(nn.Module):
    """
    GPT model for the transformer model
    This is a simple GPT model that takes in the input and returns the output of the GPT model

    """

    def __init__(self, vocab_size, emb_dim, num_heads, n_transformer_layers, dropout=0.2):
        super(GPT, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim)
        self.tranformer_layers = nn.ModuleList(
            [TransformerBlock(emb_dim, num_heads, dropout) for _ in range(n_transformer_layers)])
        self.nn_linear = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, targets=None):
        # shape of inputs is (B, T)
        X = self.embedding(inputs)  # (B, T, emb_dim)
        X = self.positional_encoding(X)  # (B, T, emb_dim)
        mask = torch.tril(torch.ones(X.shape[1], X.shape[1], dtype=torch.long), diagonal=1)  # (B, T, T)
        for layer in self.tranformer_layers:
            X = layer(X, mask)  # (B, T, emb_dim)

        X = self.dropout(X)  # (B, T, emb_dim)
        logits = self.nn_linear(X)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            # shape of targets is (B, T)
            B, T, C = logits.shape  # (B, T, vocab_size)
            logits = logits.view(B * T, C)  # (B*T, vocab_size)
            targets = targets.view(B * T)  # (B*T)
            loss = F.cross_entropy(logits, targets)  # scalar
        return logits, loss

    def generate(self, inputs, max_new_tokens=128):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(inputs)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            inputs = torch.cat((inputs, index_next), dim=1)
        return inputs
