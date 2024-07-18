import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from tinymm.utils import create_timm_model
from tinymm.model_config import ModelConfig, CLIPBaseConfig
from tinymm.GPT.model import GPTConfig, GPT


class ImageEncoder(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        base_model = create_timm_model(config)
        layers = list(base_model.children())
        if hasattr(layers[-1], "fc"):
            layers[-1].fc = nn.Linear(layers[-1].fc.in_features, config.clip_n_embd)
            self.last_proj = None
        else:
            layers = layers[:-1]
            self.last_proj = nn.Linear(config.image_embd, config.clip_n_embd)
        self.encoder = nn.Sequential(*layers)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        if self.last_proj:
            out = self.encoder(inp)
            last_token = out[:, -1, :]
            return self.last_proj(last_token)
        return self.encoder(inp)


class TextEncoder(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        gconfig = GPTConfig(config)
        gconfig.is_causal = False  # Use bidirectional attention
        self.encoder = GPT(gconfig)
        self.txt_proj = nn.Linear(gconfig.n_embd, config.clip_n_embd)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        out = self.encoder(inp)
        out = out[:, -1, :]
        return self.txt_proj(out)


class CLIP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.img_encoder = ImageEncoder(config)
        self.txt_encoder = TextEncoder(config)
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
    config = CLIPBaseConfig()
    clip = CLIP(config)

    imgs = torch.rand(8, 3, 256, 256)
    txts = torch.randint(0, 50304, (8, 64))
    loss = clip((imgs, txts))
    print(loss)
