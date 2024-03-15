import torch

from torch.utils import data
from CLIP.dataset import CC3MList, CC3MDataset
from CLIP.model import ImageConfig, GPTConfig, CLIP


class CLIPProvider:
    def __init__(self, config):
        self.config = config
        # prepare dataset
        lst = CC3MList(config.data_path, config.eval_ratio)
        train_ds = CC3MDataset(lst.to_train_list(), config.seq_len)
        eval_ds = CC3MDataset(lst.to_eval_list(), config.seq_len)

        self.train_loader = data.DataLoader(
            train_ds,
            config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        self.train_batch_iter = iter(self.train_loader)

        self.val_loader = data.DataLoader(
            eval_ds,
            config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def train_step(self, model, ctx):
        try:
            images, texts = next(self.train_batch_iter)
            if len(images) < self.config.batch_size:
                self.train_batch_iter = iter(self.train_loader)
                images, texts = next(self.train_batch_iter)
        except StopIteration:
            self.train_batch_iter = iter(self.train_loader)
            images, texts = next(self.train_batch_iter)

        images = images.cuda().permute(0, 3, 1, 2)
        texts = texts.cuda()

        with ctx:
            logits_image, logits_text, loss = model((images, texts))

        return logits_image, logits_text, loss  # train_result

    def get_metrics(self, train_result, device_type, iteration):
        """What 'train_step' output, is what 'log' get as input"""
        logits_image, _, loss = train_result
        _, predict = torch.max(logits_image, dim=-1)
        correct_labels = torch.arange(logits_image.size(0), device=device_type)
        correct = predict == correct_labels
        accuracy = correct.sum().item() / correct.size(0)
        return iteration // len(self.train_loader), accuracy, loss

    @staticmethod
    def construct_model(config):
        tconfig = GPTConfig()
        tconfig.seq_len = config.seq_len
        return CLIP(ImageConfig(), tconfig)

    def validate(self, model, ctx, device_type):
        total_loss = 0.0
        batch_iter = iter(self.val_loader)
        sum_accuracy = 0
        length = len(self.val_loader)
        for _ in range(length - 1):
            images, texts = next(batch_iter)
            images = images.cuda().permute(0, 3, 1, 2)
            texts = texts.cuda()
            # forward
            with ctx:
                logits_image, _, loss = model((images, texts))
            # accuracy
            _, predict = torch.max(logits_image, dim=-1)
            correct_labels = torch.arange(logits_image.size(0), device=device_type)
            correct = predict == correct_labels
            sum_accuracy += correct.sum().item() / correct.size(0)
            # loss
            total_loss += loss.item()
        return total_loss / length, sum_accuracy / length
