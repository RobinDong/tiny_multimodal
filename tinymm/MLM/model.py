import torch.nn.functional as F

from dataclasses import dataclass
from torch import nn
from tinymm.model_config import ModelConfig
from tinymm.GPT.model import GPT, GPTConfig

@dataclass
class MLMConfig(ModelConfig):
    model_name: str = "MLM"
    batch_size: int = 128

class MLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        gconfig = GPTConfig()
        self.encoder = GPT(gconfig)
        print("MLM model number of parameters:", self.get_num_params())

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        texts, targets = inp
        out = self.encoder(texts)  # B, S, E
        logits = self.encoder.lm_head(out)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return logits, loss
