from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str


@dataclass
class TrainConfig:
    data_path: tuple = (
        "/home/robin/Downloads/cc12m",
        "/home/robin/Downloads/cc3m",
        "/home/robin/Downloads/sbu_caption",
    )
    eval_ratio: float = 0.05
    num_workers: int = 4
    lr: float = 1e-5
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
class CLIPConfig(ModelConfig):
    model_name: str = "CLIP"
    batch_size: int = 64
    image_encoder_name: str = "convnextv2_tiny"
    image_dropout: float = 0.0
    text_encoder_name: str = "GPT"
    text_embd: int = 512
    text_layer: int = 30
    text_head: int = 8
    text_dropout: float = 0.0
    clip_n_embd: int = 512
