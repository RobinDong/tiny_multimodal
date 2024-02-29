import time
import torch
import torch.utils.data as data

from dataclasses import dataclass

from dataset import CC3MList, CC3MDataset
from model import ImageConfig, GPTConfig, CLIP


@dataclass
class TrainConfig:
    data_path: str = "/home/robin/Downloads/CC3M"
    eval_ratio: float = 0.1
    batch_size: int = 64
    num_workers: int = 8
    resume: bool = False
    lr: float = 1e-4
    grad_clip: float = 1.0
    seq_len: int = 64
    log_iters: int = 1000
    eval_iters: int = 10000
    max_iters: int = 1000000


class Trainer:

    def __init__(self, config):
        self.config = config
        self.device_type = "cuda"
        self.dtype = "bfloat16"
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == "float16"))
        self.ctx = torch.amp.autocast(
            device_type=self.device_type, dtype=torch.bfloat16
        )

        # prepare dataset
        lst = CC3MList(config.data_path, 0.1)
        train_ds = CC3MDataset(lst.to_train_list(), config.seq_len)
        eval_ds = CC3MDataset(lst.to_eval_list(), config.seq_len)

        self.train_loader = data.DataLoader(
            train_ds,
            config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            pin_memory=True,
        )

        self.eval_loader = data.DataLoader(
            eval_ds,
            config.batch_size // 8,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def train_loop(self, model, optimizer, batch_iter):
        try:
            images, texts = next(batch_iter)
        except StopIteration:
            batch_iter = iter(self.train_loader)
            images, texts = next(batch_iter)
        except Exception as ex:
            print("Loading data exception:", ex)

        images = images.cuda().permute(0, 3, 1, 2)
        texts = texts.cuda()

        with self.ctx:
            loss = model((images, texts))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return loss

    def train(self):
        iconfig = ImageConfig()
        tconfig = GPTConfig()
        tconfig.seq_len = self.config.seq_len

        model = CLIP(iconfig, tconfig).cuda()
        model = torch.compile(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=0.0,
            amsgrad=True,
        )
        batch_iter = iter(self.train_loader)
        begin = time.time()

        for iteration in range(self.config.max_iters):
            loss = self.train_loop(model, optimizer, batch_iter)

            if iteration % self.config.log_iters == 0:
                now = time.time()
                duration = now - begin
                begin = now
                print(f"[{iteration:03d}] loss: {loss.item():.4f} time {duration:.4f}")
            if iteration % self.config.eval_iters == 0:
                print("eval")


if __name__ == "__main__":
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()
