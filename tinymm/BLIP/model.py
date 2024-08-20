import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from transformers import BertTokenizerFast
from tinymm.utils import create_timm_model
from tinymm.model_config import ModelConfig, BLIPBaseConfig
from tinymm.GPT.model import GPTConfig, GPT, Block

MASKED_RATIO = 0.15


class ImageEncoder(nn.Module):
    vit_output_dims: int = 768

    def __init__(self, config: ModelConfig):
        super().__init__()

        base_model = create_timm_model(config)
        layers = list(base_model.children())[:-1]
        self.encoder = nn.Sequential(*layers)
        self.img_proj = nn.Linear(self.vit_output_dims, config.itc_embd)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        out = self.encoder(inp)
        last_token = out[:, -1, :]
        return self.img_proj(last_token), out


class TextEncoder(nn.Module):
    def __init__(self, gconfig, config):
        super().__init__()
        self.encoder = GPT(gconfig)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, inp):
        return self.encoder(inp)
        last_token = out[:, 0, :]
        return self.txt_proj(last_token), out


class BLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        gconfig = GPTConfig(config)
        gconfig.is_causal = False  # Use bidirectional attention
        self.img_encoder = ImageEncoder(config)
        self.txt_encoder = TextEncoder(
            gconfig, config
        )  # txt_encoder is also the multi-modal model (BLIP)
        print("Image Encoder number of parameters:", self.img_encoder.get_num_params())
        print("Text Encoder number of parameters:", self.txt_encoder.get_num_params())

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.multimodal_encoder = nn.ModuleList(
            [Block(gconfig) for _ in range(config.multimodal_layer)]
        )
        self.itm_mlp = nn.Linear(config.text_embd, 2)
        self.txt_proj = nn.Linear(config.text_embd, config.itc_embd)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            "google-bert/bert-base-uncased"
        )
        self.tokenizer.add_tokens(["[Encode]", "[Decode]"])

    def pick_negative_samples(self, logits_per_image, txt_feature, batch_size):
        # find negative text for each image
        image_matrix = F.softmax(logits_per_image, dim=-1)
        image_matrix.fill_diagonal_(float("inf"))
        txt_feature_neg = []
        for index in range(batch_size):
            row = 1.0 / image_matrix[index]
            neg_idx = torch.multinomial(row, 1).item()
            assert neg_idx != index
            txt_feature_neg.append(txt_feature[neg_idx])
        return torch.stack(txt_feature_neg, dim=0)

    def trunc_texts(self, ids):
        targets = ids.clone()
        ids[:, 0] = self.tokenizer.encode("[Decode]")[1]
        ids = torch.tril(ids)
        return ids, targets

    def forward(self, inp, inference=False):
        images, texts = inp
        batch_size, text_seq_len = texts.size()
        itm_input = texts.clone()
        itm_input[:, 0] = self.tokenizer.encode("[Encode]")[1]

        # ITC loss
        img_embds, img_feature = self.img_encoder(images)
        out = self.txt_encoder(texts)
        for block in self.multimodal_encoder:
            out = block(out)
        last_token = out[:, 0, :]
        txt_embds = self.txt_proj(last_token)
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

        txt_feature = self.txt_encoder(itm_input)
        if inference:
            img_feature_all = img_feature
            txt_feature_all = txt_feature
        else:
            txt_feature_neg = self.pick_negative_samples(
                logits_per_image, txt_feature, batch_size
            )
            # concat postive and negative samples
            img_feature_all = torch.cat((img_feature, img_feature), dim=0)
            txt_feature_all = torch.cat((txt_feature, txt_feature_neg), dim=0)
        out = torch.cat((img_feature_all, txt_feature_all), dim=1)
        for block in self.multimodal_encoder:
            out = block(out)  # B, S, E
        # ITM loss
        if inference:
            itm_labels = torch.tensor([1] * batch_size)
        else:
            itm_labels = torch.tensor(
                [1] * batch_size + [0] * batch_size, device=images.device
            )
        cls_token = out[:, -text_seq_len, :]
        itm_out = self.itm_mlp(cls_token)  # B, S, 2
        itm_loss = F.cross_entropy(itm_out, itm_labels)
        # LM loss
        if inference:
            return self.txt_encoder.encoder.lm_head(
                out[:batch_size, -text_seq_len + 1 :, :]
            )
        lm_input, targets = self.trunc_texts(texts)
        txt_feature = self.txt_encoder(lm_input)
        out = torch.cat((img_feature, txt_feature), dim=1)
        for block in self.multimodal_encoder:
            block.attn.config.is_causal = True
            out = block(out)  # B, S, E
            block.attn.config.is_causal = False
        logits = self.txt_encoder.encoder.lm_head(
            out[:batch_size, -text_seq_len + 1 :, :]
        )
        targets = targets[:batch_size, -text_seq_len + 1 :].contiguous()
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return (
            logits_per_image,
            logits_per_text,
            labels,
            logits,
            targets,
            itm_out,
            itm_labels,
            itc_loss,
            itm_loss,
            lm_loss,
            itc_loss + itm_loss + lm_loss,
        )


if __name__ == "__main__":
    config = BLIPBaseConfig()
    encoder = ImageEncoder(config)
    image = torch.rand([64, 3, 256, 256])
    tok, out = encoder(image)
    print("tok:", tok.size(), "out:", out.size())

    model = BLIP(config)
    text = torch.rand([64, 64]) * 100
    print("text:", text.long())
    _, _, _, _, _, loss = model((image, text.long(), text.long()))
    print("loss:", loss)
