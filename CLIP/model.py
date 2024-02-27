import timm
import torch
import torch.nn as nn

from dataclasses import dataclass


@dataclass
class CLIPConfig:
    image_encoder_name: str = "convnextv2_nano"
    image_dropout: float = 0.2


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        base_model = timm.create_model(
            config.image_encoder_name,
            pretrained=True,
            in_chans=3,
            drop_rate=config.image_dropout,
            drop_path_rate=config.image_dropout,
        )
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        print(self.encoder)

    def forward(self, inp):
        return self.encoder(inp)


config = CLIPConfig()
img_enc = ImageEncoder(config)
img = torch.rand(1, 3, 256, 256)
img_embd = img_enc(img)
print(img_embd.size())
