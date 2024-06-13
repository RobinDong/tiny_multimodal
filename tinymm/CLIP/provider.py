import torch

from collections import OrderedDict
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
    def extract_data_entry(data_entry):
        images, texts = data_entry
        return images.cuda().permute(0, 3, 1, 2), texts.cuda()

    @staticmethod
    def train_step(model, data_entry, ctx):
        images, texts = CLIPProvider.extract_data_entry(data_entry)

        with ctx:
            logits_image, logits_text, loss = model((images, texts))

        return logits_image, logits_text, loss  # train_result

    @staticmethod
    def get_accuracy(out, target):
        _, predict = torch.max(out, dim=-1)
        correct = predict == target
        accuracy = correct.sum().item() / correct.size(0)
        return accuracy

    @staticmethod
    def get_metrics(train_result, device_type, train_loader):
        """What 'train_step' output, is what 'log' get as input"""
        logits_image, logits_text, loss = train_result
        correct_labels = torch.arange(logits_image.size(0), device=device_type)
        return OrderedDict(
            [
                ("loss", loss.item()),
                ("img_accu", CLIPProvider.get_accuracy(logits_image, correct_labels)),
                ("txt_accu", CLIPProvider.get_accuracy(logits_text, correct_labels)),
            ]
        )

    @staticmethod
    def construct_model(config):
        return CLIP(config.model_config)

    @staticmethod
    def get_validation_metrics(data_entry, model, ctx, device_type):
        images, texts = CLIPProvider.extract_data_entry(data_entry)
        # forward
        with ctx:
            logits_image, logits_text, loss = model((images, texts))
        # accuracy
        correct_labels = torch.arange(logits_image.size(0), device=device_type)
        image_accuracy = CLIPProvider.get_accuracy(logits_image, correct_labels)
        text_accuracy = CLIPProvider.get_accuracy(logits_text, correct_labels)
        return OrderedDict(
            [
                ("loss", loss.item()),
                ("img_accu", image_accuracy),
                ("txt_accu", text_accuracy),
                ("accuracy", (image_accuracy + text_accuracy) / 2.0),
            ]
        )
