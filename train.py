import os
import fire
import time
import math
from collections import defaultdict
from importlib import import_module
from dataclasses import asdict

import torch
from torch.utils import data
from tinymm.model_config import TrainConfig

ckpt_dir = "out"


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device_type = "cuda"
        self.dtype = "bfloat16"
        enabled = self.dtype == "bfloat16"
        self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)
        self.ctx = torch.amp.autocast(
            device_type=self.device_type, dtype=torch.bfloat16
        )
        self.train_batch_iter = None
        self.train_provider = None
        self.train_loader = self.val_loader = None

    def train_loop(self, model, optimizer):
        try:
            data_entry = next(self.train_batch_iter)
            if len(data_entry[0]) < self.config.model_config.batch_size:
                self.train_batch_iter = iter(self.train_loader)
                data_entry = next(self.train_batch_iter)
        except StopIteration:
            self.train_batch_iter = iter(self.train_loader)
            data_entry = next(self.train_batch_iter)

        train_result = self.train_provider.train_step(model, data_entry, self.ctx)
        loss = train_result[-1]

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return train_result

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

    @torch.no_grad()
    def validate(self, cmodel):
        cmodel.eval()

        batch_iter = iter(self.val_loader)
        accumulator = defaultdict(float)
        length = len(self.val_loader)
        for _ in range(length - 1):
            data_entry = next(batch_iter)
            metrics = self.train_provider.get_validation_metrics(
                data_entry, cmodel, self.ctx, self.device_type
            )
            for key, val in metrics.items():
                accumulator[key] += val

        for key, val in accumulator.items():
            accumulator[key] /= length

        cmodel.train()
        return accumulator

    def init(self, resume: str, provider: str):
        if resume:
            checkpoint = torch.load(resume, map_location=self.device_type)
            state_dict = checkpoint["model"]
            self.config = TrainConfig(**checkpoint["train_config"])
            iter_start = checkpoint["iteration"] + 1
            module = import_module("tinymm.model_config")
            class_ = getattr(module, f"{self.config.model_config['model_name']}Config")
            self.config.model_config = class_(**self.config.model_config)
            model_name = self.config.model_config.model_name
            module = import_module(f"tinymm.{model_name}.provider")
            class_ = getattr(module, f"{model_name}Provider")
            self.train_provider = class_(self.config.model_config)
            model = self.train_provider.construct_model(self.config).cuda()
            model.load_state_dict(state_dict)
            print(f"Resume from {iter_start - 1} for model {model_name}...")
        else:
            iter_start = 1
            module = import_module(f"tinymm.{provider}.model")
            class_ = getattr(module, f"{provider}Config")
            self.config.model_config = class_()
            module = import_module(f"tinymm.{provider}.provider")
            class_ = getattr(module, f"{provider}Provider")
            self.train_provider = class_(self.config)
            model = self.train_provider.construct_model(self.config).cuda()

        train_ds, eval_ds = self.train_provider.get_datasets(config)
        self.train_loader = data.DataLoader(
            train_ds,
            self.config.model_config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        self.train_batch_iter = iter(self.train_loader)

        self.val_loader = data.DataLoader(
            eval_ds,
            self.config.model_config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        return model, iter_start

    def train(self, resume="", provider="CLIP", learning_rate=None):
        model, iter_start = self.init(resume, provider)
        if learning_rate:
            self.config.lr = learning_rate
        cmodel = torch.compile(model)
        optimizer = torch.optim.AdamW(
            cmodel.parameters(),
            lr=self.config.lr,
            weight_decay=0.0,
            amsgrad=True,
        )
        best_val_accuracy = 1e-9
        begin = time.time()

        for iteration in range(iter_start, self.config.max_iters):
            lr = self.get_lr(iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            train_result = self.train_loop(cmodel, optimizer)

            if iteration % self.config.log_iters == 0 and iteration > 0:
                metrics = self.train_provider.get_metrics(
                    train_result, self.device_type, self.train_loader
                )
                epoch = iteration // len(self.train_loader)
                now = time.time()
                duration = now - begin
                begin = now
                messages = [f"[{epoch:03d}: {iteration:06d}]"]
                for name, val in metrics.items():
                    messages.append(f"{name}: {val:.3f}")
                messages.append(f"lr: {lr:.3e}")
                messages.append(f"time: {duration:.1f}")
                print(" ".join(messages))
            if iteration % self.config.eval_iters == 0 and iteration > 0:
                accumulator = self.validate(cmodel)
                avg_accuracy = accumulator["accuracy"]
                if avg_accuracy > best_val_accuracy:
                    checkpoint = {
                        "model": model.state_dict(),
                        "iteration": iteration,
                        "train_config": asdict(self.config),
                        "eval_accuracy": avg_accuracy,
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(
                            ckpt_dir,
                            f"{self.config.model_config.model_name}_{iteration}.pt",
                        ),
                    )
                messages = ["[Val]"]
                for name, val in metrics.items():
                    messages.append(f"{name}: {val:.3f}")
                print(" ".join(messages))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    config = TrainConfig()
    trainer = Trainer(config)

    fire.Fire(trainer.train)
