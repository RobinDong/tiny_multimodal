import torch
import torch.nn.functional as F

from collections import OrderedDict

from tinymm.CLIP.dataset import CC3MList
from tinymm.ALBEF.dataset import ALBEFDataset
from tinymm.BLIP.model import BLIP


class BLIPProvider:
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
    def extract_data_entry(data_entry):
        images, texts = data_entry
        return images.cuda().permute(0, 3, 1, 2), texts.cuda()

    @staticmethod
    def train_step(model, data_entry, ctx):
        images, texts = BLIPProvider.extract_data_entry(data_entry)

        with ctx:
            train_result = model((images, texts))

        return train_result

    @staticmethod
    def get_accuracy(out, target):
        _, predict = torch.max(out, dim=-1)
        correct = predict == target
        accuracy = correct.sum().item() / correct.size(0)
        return accuracy

    @staticmethod
    def get_metrics(train_result, device_type, train_loader):
        """What 'train_step' output, is what 'log' get as input"""
        (
            logits_image,
            logits_text,
            itc_labels,
            logits,
            targets,
            itm_out,
            match_labels,
            itc_loss,
            itm_loss,
            lm_loss,
            _,
        ) = train_result

        batch_size, seq_len, _ = logits.size()
        logits = logits.view(batch_size * seq_len, -1)
        return OrderedDict(
            [
                ("itc_loss", itc_loss.item()),
                ("itm_loss", itm_loss.item()),
                ("lm_loss", lm_loss.item()),
                ("img_accu", BLIPProvider.get_accuracy(logits_image, itc_labels)),
                ("txt_accu", BLIPProvider.get_accuracy(logits_text, itc_labels)),
                ("itm_accu", BLIPProvider.get_accuracy(itm_out, match_labels)),
                ("lm_accu", BLIPProvider.get_accuracy(logits, targets.view(-1))),
            ]
        )

    @staticmethod
    def construct_model(config):
        return BLIP(config.model_config)

    @staticmethod
    def get_validation_metrics(data_entry, model, ctx, device_type):
        images, texts = BLIPProvider.extract_data_entry(data_entry)
        # forward
        with ctx:
            (
                logits_image,
                logits_text,
                itc_labels,
                logits,
                targets,
                itm_out,
                match_labels,
                _,
                _,
                _,
                loss,
            ) = model((images, texts))
        # accuracy
        batch_size, seq_len, _ = logits.size()
        logits = logits.view(batch_size * seq_len, -1)
        targets = targets.view(-1)
        lm_accuracy = BLIPProvider.get_accuracy(logits, targets)
        lm_loss = F.cross_entropy(logits, targets)
        return OrderedDict(
            [
                ("loss", loss.item()),
                ("lm_loss", lm_loss.item()),
                ("img_accu", BLIPProvider.get_accuracy(logits_image, itc_labels)),
                ("txt_accu", BLIPProvider.get_accuracy(logits_text, itc_labels)),
                ("itm_accu", BLIPProvider.get_accuracy(itm_out, match_labels)),
                ("lm_accu", lm_accuracy),
                ("accuracy", lm_accuracy),
            ]
        )
