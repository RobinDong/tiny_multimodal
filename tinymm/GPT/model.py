import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from tinymm.model_config import ModelConfig


@dataclass
class GPTConfig:
    def __init__(self, config: ModelConfig = None):
        if config:
            self.n_embd = config.text_embd
            self.n_layer = config.text_layer
            self.n_head = config.text_head
            self.dropout = config.text_dropout

    vocab_size: int = 50304
    seq_len: int = 64
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.head_dim = self.config.n_embd // self.config.n_head
        self.div = math.sqrt(self.head_dim)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(1, 1, self.config.seq_len, self.config.seq_len)),
            )
        self.attn_drop = nn.Dropout(self.config.dropout)
        self.resid_drop = nn.Dropout(self.config.dropout)

    def forward(self, inp):
        batch_size, seq_len, _ = inp.size()
        query, key, value = self.c_attn(inp).split(self.config.n_embd, dim=-1)
        # (B, nh, S, hd)
        query = query.view(
            batch_size, seq_len, self.config.n_head, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, seq_len, self.config.n_head, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, seq_len, self.config.n_head, self.head_dim
        ).transpose(1, 2)

        if self.flash:
            attn = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            attn = query @ key.transpose(-2, -1) / self.div
            attn = attn.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )  # (B, nh, S, S)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn = attn @ value  # (B, S, nh, hs)

        out = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        out = self.c_proj(out)
        return self.resid_drop(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, inp):
        inp = self.c_fc(inp)
        inp = self.gelu(inp)
        inp = self.c_proj(inp)
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
        inp = inp + self.attn(inp)
        inp = self.ln2(inp)
        return inp + self.mlp(inp)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.seq_len, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "hidden": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, pa in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    pa, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        _, seq_len = idx.size()

        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        out = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.hidden:
            out = block(out)
        out = self.transformer.ln_f(out)
        return out


if __name__ == "__main__":
    config = GPTConfig()
    gpt = GPT(config)
    sample = torch.randint(0, 512, (32, 512))
    out = gpt(sample)
    print(out.view(out.size()[0], -1).size())
