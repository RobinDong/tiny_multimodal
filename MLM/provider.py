import torch

from MLM.dataset import ENWikiList, ENWikiDataset
from MLM.model import MLM
from GPT.model import GPTConfig


class MLMProvider:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_datasets(config):
        # prepare dataset
        lst = ENWikiList("enwiki/", config.eval_ratio)
        train_ds = ENWikiDataset(lst.to_train_list(), config.seq_len)
        eval_ds = ENWikiDataset(lst.to_eval_list(), config.seq_len)
        return train_ds, eval_ds

    @staticmethod
    def train_step(model, data_entry, ctx):
        texts, targets = data_entry
        texts = texts.cuda()
        targets = targets.cuda()

        with ctx:
            logits, loss = model((texts, targets))

        return (
            logits,
            targets,
            loss,
        )  # train_result. The train_result[-1] must be loss

    @staticmethod
    def get_metrics(
        train_result, device_type, iteration, train_loader
    ):  # pylint: disable=unused-argument
        """What 'train_step' output, is what 'log' get as input"""
        logits, targets, loss = train_result

        _, predict = torch.max(logits, dim=-1)
        correct = predict == targets
        accuracy = correct.sum().item() / correct.size(0) / correct.size(1)
        return iteration // len(train_loader), accuracy, loss

    @staticmethod
    def construct_model(config):
        tconfig = GPTConfig(seq_len=config.seq_len)
        tconfig.n_layer = 12
        tconfig.n_head = 12
        tconfig.n_embd = 768
        return MLM(tconfig)

    @staticmethod
    def get_validate_accuracy(
        data_entry, model, ctx, device_type
    ):  # pylint: disable=unused-argument
        texts, targets = data_entry
        texts = texts.cuda()
        targets = targets.cuda()
        # forward
        with ctx:
            logits, loss = model((texts, targets))
        # accuracy
        _, predict = torch.max(logits, dim=-1)
        correct = predict == targets
        accuracy = correct.sum().item() / correct.size(0) / correct.size(1)
        return accuracy, loss.item()
