import numpy as np

from tinymm.CLIP.dataset import CC3MList, CC3MDataset

MASK_ID = 50257
MASKED_RATIO = 0.15


class ALBEFDataset(CC3MDataset):
    def __getitem__(self, index):
        images, ids = super().get_raw(index)
        length = len(ids)
        # randomly set 15% MASK
        increase = np.arange(length)
        n_choices = int(length * MASKED_RATIO)
        indexes = np.sort(np.random.choice(increase, n_choices))
        targets = ids.copy()
        length = len(targets)
        if length > self.seq_len:
            targets = targets[: self.seq_len]
        elif length < self.seq_len:
            targets = np.pad(targets, (0, (self.seq_len - length)), "constant")
        ids[indexes] = MASK_ID

        images, ids = super().raw_to_ds(images, ids)
        return images, ids, targets


if __name__ == "__main__":
    lst = CC3MList(("/home/robin/Downloads/cc3m",), 0.1)
    ds = ALBEFDataset(lst.id_to_filename, lst.to_eval_list(), 64)
    length = len(ds)
    for index in range(length):
        image, txt, tgt = ds[index]
        assert len(txt) == 64
        assert len(tgt) == 64
        print(txt, tgt)
        print("\n")
