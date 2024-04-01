import torch

from tinymm.CLIP.dataset import CC3MList, CC3MDataset
from tinymm.CLIP.model import CLIP


class CLIPProvider:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_datasets(config):
        # prepare dataset
        lst = CC3MList(config.data_path, config.eval_ratio)
        train_ds = CC3MDataset(lst.id_to_filename, lst.to_train_list(), config.seq_len)
        eval_ds = CC3MDataset(lst.id_to_filename, lst.to_eval_list(), config.seq_len)
        return train_ds, eval_ds

    @staticmethod
    def train_step(model, data_entry, ctx):
        images, texts = data_entry
        images = images.cuda().permute(0, 3, 1, 2)
        texts = texts.cuda()

        with ctx:
            logits_image, logits_text, loss = model((images, texts))

        return logits_image, logits_text, loss  # train_result

    @staticmethod
    def get_metrics(train_result, device_type, iteration, train_loader):
        """What 'train_step' output, is what 'log' get as input"""
        logits_image, _, loss = train_result
        _, predict = torch.max(logits_image, dim=-1)
        correct_labels = torch.arange(logits_image.size(0), device=device_type)
        correct = predict == correct_labels
        accuracy = correct.sum().item() / correct.size(0)
        return iteration // len(train_loader), accuracy, loss

    @staticmethod
    def construct_model(config):
        return CLIP(config.model_config)

    @staticmethod
    def get_validate_accuracy(data_entry, model, ctx, device_type):
        images, texts = data_entry
        images = images.cuda().permute(0, 3, 1, 2)
        texts = texts.cuda()
        # forward
        with ctx:
            logits_image, _, loss = model((images, texts))
        # accuracy
        _, predict = torch.max(logits_image, dim=-1)
        correct_labels = torch.arange(logits_image.size(0), device=device_type)
        correct = predict == correct_labels
        return correct.sum().item() / correct.size(0), loss.item()
