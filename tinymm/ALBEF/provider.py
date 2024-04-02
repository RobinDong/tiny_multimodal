import torch

from tinymm.CLIP.dataset import CC3MList
from tinymm.ALBEF.dataset import ALBEFDataset
from tinymm.ALBEF.model import ALBEF


class ALBEFProvider:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_datasets(config):
        # prepare dataset
        lst = CC3MList(config.data_path, config.eval_ratio)
        train_ds = ALBEFDataset(lst.id_to_filename, lst.to_train_list(), config.seq_len)
        eval_ds = ALBEFDataset(lst.id_to_filename, lst.to_eval_list(), config.seq_len)
        return train_ds, eval_ds

    @staticmethod
    def train_step(model, data_entry, ctx):
        images, texts, targets = data_entry
        images = images.cuda().permute(0, 3, 1, 2)
        texts = texts.cuda()
        targets = targets.cuda()

        with ctx:
            (
                logits_image,
                logits_text,
                labels,
                logits,
                targets,
                loss,
            ) = model((images, texts, targets))

        return (
            logits_image,
            logits_text,
            labels,
            logits,
            targets,
            loss,
        )  # train_result

    @staticmethod
    def get_accuracy(out, target):
        _, predict = torch.max(out, dim=-1)
        correct = predict == target
        accuracy = correct.sum().item() / correct.size(0)
        return accuracy

    @staticmethod
    def get_metrics(train_result, device_type, iteration, train_loader):
        """What 'train_step' output, is what 'log' get as input"""
        (
            logits_image,
            logits_text,
            labels,
            logits,
            targets,
            loss,
        ) = train_result

        text_accu = ALBEFProvider.get_accuracy(logits_text, labels)
        batch_size, seq_len, _ = logits.size()
        logits = logits.view(batch_size * seq_len, -1)
        mlm_accu = ALBEFProvider.get_accuracy(logits, targets.view(-1))
        print(f"text accu: {text_accu:.4f}, mlm accu: {mlm_accu:.4f}")
        return (
            iteration // len(train_loader),
            ALBEFProvider.get_accuracy(logits_image, labels),
            loss,
        )

    @staticmethod
    def construct_model(config):
        return ALBEF(config.model_config)

    @staticmethod
    def get_validate_accuracy(data_entry, model, ctx, device_type):
        images, texts, targets = data_entry
        images = images.cuda().permute(0, 3, 1, 2)
        texts = texts.cuda()
        targets = targets.cuda()
        # forward
        with ctx:
            _, _, _, logits, targets, loss = model((images, texts, targets))
        # accuracy
        batch_size, seq_len, _ = logits.size()
        logits = logits.view(batch_size * seq_len, -1)
        accuracy = ALBEFProvider.get_accuracy(logits, targets.view(-1))
        return accuracy, loss.item()
