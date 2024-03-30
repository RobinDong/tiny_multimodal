from dataclasses import dataclass

import timm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tinymm.GPT.model import GPTConfig, GPT

clip_n_embd = 512


@dataclass
class ImageConfig:
    image_encoder_name: str = "convnextv2_tiny"
    image_dropout: float = 0.0


class ImageEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        base_model = timm.create_model(
            config.image_encoder_name,
            pretrained=False,
            in_chans=3,
            drop_rate=config.image_dropout,
            drop_path_rate=config.image_dropout,
        )
        layers = list(base_model.children())
        layers[-1].fc = nn.Linear(layers[-1].fc.in_features, clip_n_embd)
        self.encoder = nn.Sequential(*layers)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        return self.encoder(inp)


class TextEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = GPT(config).cuda()
        self.txt_proj = nn.Linear(config.n_embd, clip_n_embd)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        out = self.encoder(inp)
        out = out[:, -1, :]
        # out = out.contiguous().view(inp.size(0), -1)
        return self.txt_proj(out)


class CLIP(nn.Module):

    def __init__(self, img_config, txt_config):
        super().__init__()

        self.img_encoder = ImageEncoder(img_config).cuda()
        self.txt_encoder = TextEncoder(txt_config).cuda()
        print("Image Encoder number of parameters:", self.img_encoder.get_num_params())
        print("Text Encoder number of parameters:", self.txt_encoder.get_num_params())

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, inp):
        images, txts = inp
        img_f = self.img_encoder(images)
        txt_f = self.txt_encoder(txts)
        img_embds = F.normalize(img_f, dim=-1)  # (B, E)
        txt_embds = F.normalize(txt_f, dim=-1)  # (B, E)

        # mainly learned from https://github.com/openai/CLIP/blob/main/clip/model.py
        logits_per_image = self.logit_scale.exp() * img_embds @ txt_embds.T
        logits_per_text = logits_per_image.T

        labels = torch.arange(logits_per_image.size(0), device=images.device)
        loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2.0
        return logits_per_image, logits_per_text, loss


if __name__ == "__main__":
    iconfig = ImageConfig()
    gconfig = GPTConfig()

    clip = CLIP(iconfig, gconfig)

    imgs = torch.rand(8, 3, 256, 256)
    txts = torch.randint(0, 50304, (8, 512))
    loss = clip((imgs, txts))
    print(loss)
