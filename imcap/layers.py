import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from .utils import *


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.scale = emb_size**0.5

    def forward(self, tokens):
        return self.embedding(tokens.long()) * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dp_rate, maxlen = 1_000):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10_000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dp_rate)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class ProjectionHead(nn.Module):
    def __init__(self, d_model=512, dp_rate=0.1):
        super().__init__()
        self.proj = nn.LazyLinear(d_model)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dp_rate)
        self.dense = nn.LazyLinear(d_model)
        self.dropout2 = nn.Dropout(dp_rate)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):   # (..., features)
        p = self.proj(x)
        x = self.dropout1(self.activation(p))
        x = self.dropout2(self.dense(x))
        x = self.ln(x + p)
        return x


class CaptionModel(nn.Module):
    def __init__(self, encoder, vocab_size, num_decoder_layers=6, nheads=8, d_model=512,
                 dim_feedforward=2048, dp_rate=0.1, activation='relu', bn_eval=True):
        super().__init__()
        self.encoder = encoder
        freeze_weights(self.encoder)
        if bn_eval: set_bn_eval(self.encoder)
        
        self.projection_head = ProjectionHead(d_model, dp_rate)

        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dp_rate)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nheads, dim_feedforward, dp_rate,
                                                   activation)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.generator = nn.Linear(d_model, vocab_size)


    def encode_image(self, x):
        # Extract Image Features
        x = self.encoder(x)                 # We can precalculate the the features and store them, I just don't have enough space to store on colab, they will be 20-30+ GB.
        # (B, features, h, w)
        x = x.flatten(-2)    # flatten each feature
        # (B, features, h*w)
        x = x.permute(2,0,1)
        # (h*w, B, features)
        x = self.projection_head(x)
        # (h*w, B, d_model)
        return x


    def decode_text(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Generate Captions
        tgt = self.pos_enc(self.tok_emb(tgt))
        # (seq_len, B, d_model)
        x = self.decoder(tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # (seq_len, B, d_model)
        return x


    def forward(self, x, tgt, tgt_mask=None, tgt_key_padding_mask=None):   # x[B,C,H,W] , tgt[seq_len, B].type(long)
        x = self.encode_image(x)
        # (h*w, B, d_model)
        x = self.decode_text(tgt, x, tgt_mask, tgt_key_padding_mask)
        # (seq_len, B, d_model)
        x = self.generator(x)
        # (seq_len, B, vocab_size)
        return x