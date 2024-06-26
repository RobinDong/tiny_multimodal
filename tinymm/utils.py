import cv2
import timm
import torch

from importlib import import_module
from tinymm.model_config import TrainConfig, ModelConfig


def create_timm_model(config: ModelConfig):
    return timm.create_model(
        config.image_encoder_name,
        pretrained=False,
        in_chans=3,
        drop_rate=config.image_dropout,
        drop_path_rate=config.image_dropout,
    )

def load_from_checkpoint(checkpoint: str):
    checkpoint = torch.load(checkpoint, map_location="cpu")
    if "quantization" in checkpoint:
        model = checkpoint["model"]
    else:
        state_dict = checkpoint["model"]
        config = TrainConfig(**checkpoint["train_config"])
        module = import_module("tinymm.model_config")
        mconfig = config.model_config
        class_ = getattr(module, f"{mconfig['model_name']}{mconfig['model_size']}Config")
        config.model_config = class_(**config.model_config)
        model_name = config.model_config.model_name
        module = import_module(f"tinymm.{model_name}.provider")
        class_ = getattr(module, f"{model_name}Provider")
        train_provider = class_(config.model_config)
        model = train_provider.construct_model(config)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_name: str, image_size: tuple[int, int]):
    image = cv2.resize(cv2.imread(image_name), image_size).astype("float32") / 255.0
    return torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
