import torch.nn.functional as F
from torch import nn

from tinymm.GPT.model import GPT


class MLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = GPT(config).cuda()

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
