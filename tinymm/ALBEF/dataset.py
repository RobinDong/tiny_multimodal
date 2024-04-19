import torch
import numpy as np

from tinymm.CLIP.dataset import CC3MList, CC3MDataset

MASKED_RATIO = 0.15


class ALBEFDataset(CC3MDataset):
    def __getitem__(self, index):
        images, ids = super().get_raw(index)
        length = len(ids)

        # randomly set 15% MASK
        prob_arr = torch.full([length], MASKED_RATIO)
        masked_indices = torch.bernoulli(prob_arr).bool()
        masked_indices[ids == self.enc.cls_token] = False
        masked_indices[ids == self.enc.sep_token] = False

        targets = ids.copy()
        tlen = len(targets)
        if tlen > self.seq_len:
            targets = targets[: self.seq_len]
        elif tlen < self.seq_len:
            targets = np.pad(targets, (0, (self.seq_len - tlen)), "constant")
        ids[masked_indices] = self.enc.mask_token_id

        images, ids = super().raw_to_ds(images, ids)
        return images, ids, targets


if __name__ == "__main__":
    lst = CC3MList(("/home/robin/Downloads/cc3m",), 0.05)
    ds = ALBEFDataset(lst.id_to_filename, lst.to_train_list(), 64)
    length = len(ds)
    for index in range(length):
        image, ids, tgt = ds[index]
        assert len(ids) == 64
        assert len(tgt) == 64
        print(index, ids, tgt)
        print(ds.enc.decode(ids), ds.enc.decode(tgt))
        print("\n")
