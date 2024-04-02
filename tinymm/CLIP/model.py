import timm
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from torch import nn
from tinymm.model_config import ModelConfig
from tinymm.GPT.model import GPTConfig, GPT


@dataclass
class CLIPConfig(ModelConfig):
    model_name: str = "CLIP"
    batch_size: int = 64
    image_encoder_name: str = "convnextv2_tiny"
    image_dropout: float = 0.0
    text_encoder_name: str = "GPT"
    text_embd: int = 768
    text_layer: int = 12
    text_head: int = 12
    text_dropout: float = 0.0
    clip_n_embd: int = 512


class ImageEncoder(nn.Module):

    def __init__(self, config: CLIPConfig):
        super().__init__()

        base_model = timm.create_model(
            config.image_encoder_name,
            pretrained=False,
            in_chans=3,
            drop_rate=config.image_dropout,
            drop_path_rate=config.image_dropout,
        )
        layers = list(base_model.children())
        layers[-1].fc = nn.Linear(layers[-1].fc.in_features, config.clip_n_embd)
        self.encoder = nn.Sequential(*layers)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        return self.encoder(inp)


class TextEncoder(nn.Module):

    def __init__(self, config: CLIPConfig):
        super().__init__()

        gconfig = GPTConfig()
        gconfig.n_embd = config.text_embd
        gconfig.n_layer = config.text_layer
        gconfig.n_head = config.text_head
        gconfig.dropout = config.text_dropout
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

    def __init__(self, config: CLIPConfig):
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
    config = CLIPConfig()
    clip = CLIP(config)

    imgs = torch.rand(8, 3, 256, 256)
    txts = torch.randint(0, 50304, (8, 64))
    loss = clip((imgs, txts))
    print(loss)
