from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str
    model_size: str


@dataclass
class TrainConfig:
    data_path: tuple = (
        "/home/robin/Downloads/cc12m",
        "/home/robin/Downloads/cc3m",
        "/home/robin/Downloads/sbu_caption",
    )
    eval_ratio: float = 0.05
    num_workers: int = 4
    lr: float = 1e-4
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    seq_len: int = 64
    log_iters: int = 2000
    eval_iters: int = 20000
    warmup_iters: int = 4000
    lr_decay_iters: int = 512000
    max_iters: int = 1000000
    model_config: ModelConfig = None


@dataclass
class CLIPBaseConfig(ModelConfig):
    model_name: str = "CLIP"
    model_size: str = "Base"
    batch_size: int = 64
    image_encoder_name: str = "vit_base_patch16_reg4_gap_256"
    image_embd: int = 768
    image_dropout: float = 0.0
    text_encoder_name: str = "GPT"
    text_embd: int = 768
    text_layer: int = 12
    text_head: int = 12
    text_dropout: float = 0.0
    clip_n_embd: int = 768


@dataclass
class CLIPSmallConfig(ModelConfig):
    model_name: str = "CLIP"
    model_size: str = "Small"
    batch_size: int = 128
    image_encoder_name: str = "vit_medium_patch16_reg4_gap_256"
    image_embd: int = 512
    image_dropout: float = 0.0
    text_encoder_name: str = "GPT"
    text_embd: int = 512
    text_layer: int = 12
    text_head: int = 8
    text_dropout: float = 0.0
    clip_n_embd: int = 512


@dataclass
class CLIPTinyConfig(ModelConfig):
    model_name: str = "CLIP"
    model_size: str = "Tiny"
    batch_size: int = 96
    image_encoder_name: str = "convnextv2_tiny"
    image_embd: int = 384
    image_dropout: float = 0.0
    text_encoder_name: str = "GPT"
    text_embd: int = 384
    text_layer: int = 12
    text_head: int = 8
    text_dropout: float = 0.0
    clip_n_embd: int = 384


@dataclass
class ALBEFBaseConfig(ModelConfig):
    model_name: str = "ALBEF"
    model_size: str = "Base"
    batch_size: int = 32
    image_encoder_name: str = "vit_base_patch16_reg4_gap_256"
    image_dropout: float = 0.0
    text_encoder_name: str = "GPT"
    text_embd: int = 768
    text_layer: int = 6
    text_head: int = 12
    text_dropout: float = 0.0
    itc_embd: int = 256  # The original ALBEF use 256 dims for ITC loss
    multimodal_layer: int = 6
