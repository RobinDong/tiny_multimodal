import timm
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from torch import nn
from tinymm.model_config import ModelConfig
from tinymm.GPT.model import GPTConfig, GPT, Block


@dataclass
class ALBEFConfig(ModelConfig):
    model_name: str = "ALBEF"
    batch_size: int = 48
    image_encoder_name: str = "vit_medium_patch16_gap_256"
    image_dropout: float = 0.0
    text_encoder_name: str = "GPT"
    text_embd: int = 768
    text_layer: int = 6
    text_head: int = 12
    text_dropout: float = 0.0
    itc_embd: int = 128  # The original ALBEF use 256 dims for ITC loss
    multimodal_layer: int = 6


class ImageEncoder(nn.Module):
    def __init__(self, config: ALBEFConfig):
        super().__init__()

        base_model = timm.create_model(
            config.image_encoder_name,
            pretrained=True,
            in_chans=3,
            drop_rate=config.image_dropout,
            drop_path_rate=config.image_dropout,
        )
        layers = list(base_model.children())[:-1]
        self.encoder = nn.Sequential(*layers)
        self.out_proj = nn.Linear(512, config.text_embd)
        self.img_proj = nn.Linear(512, config.itc_embd)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        out = self.encoder(inp)
        last_token = out[:, -1, :]
        out = self.out_proj(out)
        return self.img_proj(last_token), out


class TextEncoder(nn.Module):

    def __init__(self, gconfig, config):
        super().__init__()
        self.encoder = GPT(gconfig)
        self.txt_proj = nn.Linear(config.text_embd, config.itc_embd)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        out = self.encoder(inp)
        last_token = out[:, -1, :]
        return self.txt_proj(last_token), out


class ALBEF(nn.Module):
    def __init__(self, config):
        super().__init__()

        gconfig = GPTConfig(config)
        self.img_encoder = ImageEncoder(config)
        self.txt_encoder = TextEncoder(gconfig, config)
        print("Image Encoder number of parameters:", self.img_encoder.get_num_params())
        print("Text Encoder number of parameters:", self.txt_encoder.get_num_params())

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.multimodal_encoder = nn.ModuleList(
            [Block(gconfig) for _ in range(config.multimodal_layer)]
        )
        self.itm_mlp = nn.Linear(config.text_embd, 2)

    def forward(self, inp):
        images, texts, targets = inp
        _, text_seq_len = texts.size()

        # ITC loss
        img_embds, img_feature = self.img_encoder(images)
        txt_embds, txt_feature = self.txt_encoder(texts)
        img_embds = F.normalize(img_embds, dim=-1)
        txt_embds = F.normalize(txt_embds, dim=-1)

        # mainly learned from https://github.com/openai/CLIP/blob/main/clip/model.py
        logits_per_image = self.logit_scale.exp() * img_embds @ txt_embds.T
        logits_per_text = logits_per_image.T

        labels = torch.arange(logits_per_image.size(0), device=images.device)
        itc_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2.0

        out = torch.cat((img_feature, txt_feature), dim=1)
        for block in self.multimodal_encoder:
            out = block(out)  # B, S, E
        # ITM loss
        # last_token = out[:, -1, :]
        # itm_out = self.itm_mlp(last_token)  # B, S, 2
        # itm_loss = F.cross_entropy(itm_out, match_labels)
        # MLM loss
        logits = self.txt_encoder.encoder.lm_head(out[:, :text_seq_len, :])
        mlm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return (
            logits_per_image,
            logits_per_text,
            labels,
            logits,
            targets,
            itc_loss + mlm_loss,
        )


if __name__ == "__main__":
    config = ALBEFConfig()
    encoder = ImageEncoder(config)
    image = torch.rand([64, 3, 256, 256])
    tok, out = encoder(image)
    print("tok:", tok.size(), "out:", out.size())

    model = ALBEF(config)
    text = torch.rand([64, 64]) * 100
    print("text:", text.long())
    _, _, _, _, _, loss = model((image, text.long(), text.long()))
    print("loss:", loss)
