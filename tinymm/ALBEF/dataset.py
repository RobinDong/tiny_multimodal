import numpy as np

from tinymm.CLIP.dataset import CC3MDataset

MASK_ID = 50257
MASKED_RATIO = 0.15


class ALBEFDataset(CC3MDataset):
    def __init__(self, id_to_filename, lst, seq_len):
        super().__init__(id_to_filename, lst, seq_len)

        self.increase = np.arange(seq_len)
        self.n_choices = int(seq_len * MASKED_RATIO)

    def __getitem__(self, index):
        images, ids = super().__getitem__(index)
        indexes = np.sort(np.random.choice(self.increase, self.n_choices))
        targets = ids.copy()
        ids[indexes] = MASK_ID
        return images, ids, targets
