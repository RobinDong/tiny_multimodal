import time
import math
import torch
import torch.utils.data as data

from dataclasses import dataclass

from dataset import CC3MList, CC3MDataset
from model import ImageConfig, GPTConfig, CLIP


@dataclass
class TrainConfig:
    data_path: str = "/home/robin/Downloads/CC3M"
    eval_ratio: float = 0.1
    batch_size: int = 128
    num_workers: int = 4
    resume: bool = False
    lr: float = 1e-4
    min_lr: float = 1e-6
    grad_clip: float = 100.0
    seq_len: int = 64
    log_iters: int = 1000
    eval_iters: int = 10000
    warmup_iters: int = 2000
    lr_decay_iters: int = 128000
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
        self.correct_labels = torch.arange(config.batch_size, device=self.device_type)

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
        self.batch_iter = iter(self.train_loader)

        self.eval_loader = data.DataLoader(
            eval_ds,
            config.batch_size // 8,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def train_loop(self, model, optimizer):
        try:
            images, texts = next(self.batch_iter)
            if len(images) < self.config.batch_size:
                self.batch_iter = iter(self.train_loader)
                images, texts = next(self.batch_iter)
        except StopIteration:
            self.batch_iter = iter(self.train_loader)
            images, texts = next(self.batch_iter)
        except Exception as ex:
            print("Loading data exception:", ex)

        images = images.cuda().permute(0, 3, 1, 2)
        texts = texts.cuda()

        with self.ctx:
            logits_image, logits_text, loss = model((images, texts))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return logits_image, logits_text, loss

    def get_lr(self, iteration):
        config = self.config
        # 1) linear warmup for warmup_iters steps
        if iteration < config.warmup_iters:
            return config.lr * iteration / config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if iteration > config.lr_decay_iters:
            return config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iteration - config.warmup_iters) / (
            config.lr_decay_iters - config.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config.min_lr + coeff * (config.lr - config.min_lr)

    def train(self):
        iconfig = ImageConfig()
        tconfig = GPTConfig()
        tconfig.seq_len = self.config.seq_len

        model = CLIP(iconfig, tconfig).cuda()
        model = torch.compile(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=0.0, amsgrad=True,
        )
        begin = time.time()

        for iteration in range(self.config.max_iters):
            lr = self.get_lr(iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            logits_image, logits_text, loss = self.train_loop(model, optimizer)

            if iteration % self.config.log_iters == 0:
                _, predict = torch.max(logits_image, dim=-1)
                correct = predict == self.correct_labels
                accuracy = correct.sum().item() / correct.size(0)
                now = time.time()
                duration = now - begin
                begin = now
                epoch = iteration // len(self.train_loader)
                print(
                    f"[{epoch:03d} : {iteration:06d}] loss: {loss.item():.4f} accu: {accuracy:.4f} lr: {lr:.4e} time: {duration:.2f}"
                )
            if iteration % self.config.eval_iters == 0:
                print("eval")


if __name__ == "__main__":
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()
