import math

import torch
import torch.nn as nn

from torch.nn import functional as F

class GPTConfig:
    seq_len: int = 512
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.2


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.head_dim = self.config.n_embd // self.config.n_head
        self.div = math.sqrt(self.head_dim)
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, self.config.seq_len, self.config.seq_len)))
        self.attn_drop = nn.Dropout(self.config.dropout)
        self.resid_drop = nn.Dropout(self.config.dropout)

    def forward(self, inp):
        batch_size, seq_len, _ = inp.size()
        query, key, value = self.c_attn(inp).split(self.config.n_embd, dim=-1)
        # (B, nh, S, hd)
        query = query.view(batch_size, seq_len, self.config.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.config.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.config.n_head, self.head_dim).transpose(1, 2)

        attn = query @ key.transpose(-2, -1) // self.div
        attn = attn.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, float("-inf"))
        attn = F.softmax(attn)
        attn = self.attn_drop(attn)
        attn = attn @ value
        out = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        out = self.c_proj(out)
        return self.resid_drop(out)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, inp):
        inp = self.linear1(inp)
        inp = self.gelu(inp)
        inp = self.linear2(inp)
        return self.drop(inp)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, inp):
        inp = self.ln1(inp)
        inp = self.attn(inp)
        inp = self.ln2(inp)
        return self.mlp(inp)


config = GPTConfig()
blk = Block(config)
sample = torch.rand(128, 512, 768)
out = blk(sample)
print(out.size())
print(sample, out)
