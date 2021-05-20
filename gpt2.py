#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


import logging
import math

logger = logging.getLogger(__name__)


class Config:
    emb_d = 128
    d = 128
    attn_drop = 0.1
    resid_drop = 0.1
    vocab = 10000
    block_size = 128
    num_h = 4
    emb_drop = 0.1
    n_layers = 10
    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)








class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d % cfg.num_h == 0

        self.K = nn.Linear(cfg.emb_d, cfg.d)
        self.Q = nn.Linear(cfg.emb_d, cfg.d)
        self.V = nn.Linear(cfg.emb_d, cfg.d)
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.resid_drop = nn.Dropout(cfg.resid_drop)
        self.proj = nn.Linear(cfg.d, cfg.d)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("causal_mask", mask)
        self.num_h = cfg.num_h

    def forward(self, x, layer_past=None):
        B, T, D = x.size()

        # B,num_h,T,hs
        k = self.K(x).view(B, T, self.num_h, D // self.num_h).transpose(1, 2)
        q = self.Q(x).view(B, T, self.num_h, D // self.num_h).transpose(1, 2)
        v = self.V(x).view(B, T, self.num_h, D // self.num_h).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # B,nh,T,T x B,nh,T,hs -> B,nh,T,hs

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.emb_d, 4 * cfg.emb_d),
            nn.GELU(),
            nn.Linear(4 * cfg.emb_d, cfg.emb_d),
            nn.Dropout(cfg.resid_drop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.emb_d)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.emb_d))
        self.drop = nn.Dropout(cfg.emb_drop)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])

        self.ln_f = nn.LayerNorm(cfg.emb_d)

        self.head = nn.Linear(cfg.emb_d, cfg.vocab, bias=False)
        self.block_size = cfg.block_size

        self.apply(self._init_weights)

        logger.info(
            "Number of parameters : %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx):
        b, t = idx.size()

        assert t <= self.block_size

        token_embeddings = self.tok_emb(idx)
        # add and drop pos embs
        x = self.drop(token_embeddings + self.pos_emb[:, :t, :])

        x = self.blocks(x)
        x = self.ln_f(x)

        # returning the logits
        return self.head(x)


def main():
    c = GPT(Config())


if __name__ == "__main__":
    main()
