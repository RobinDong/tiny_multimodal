import glob
import tiktoken

import numpy as np
from torch.utils import data

MASK_ID = 50257
MASKED_RATIO = 0.15


class ENWikiList:
    def __init__(self, path, eval_ratio):
        lines = []
        for file in glob.glob(path + "*.txt"):
            with open(file, "r", encoding="utf-8") as fp:
                lines += fp.readlines()
        np.random.shuffle(lines)
        self.lines = lines
        self.div = int(len(self.lines) * (1 - eval_ratio))

    def to_train_list(self):
        return self.lines[: self.div]

    def to_eval_list(self):
        return self.lines[self.div :]


class ENWikiDataset(data.Dataset):
    def __init__(self, lst, seq_len):
        self.lines = lst
        self.seq_len = seq_len
        # Extend tiktoken for special token "<|mask|>"
        base = tiktoken.get_encoding("gpt2")
        self.enc = tiktoken.Encoding(
            name="gpt2_mlm",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                **base._special_tokens,
                "<|mask|>": MASK_ID,
            },
        )
        self.increase = np.arange(seq_len)
        self.n_choices = int(seq_len * MASKED_RATIO)

    def __getitem__(self, index):
        line = self.lines[index]
        ids = self.enc.encode_ordinary(line)
        ids = np.array(ids, dtype=np.int64)
        length = len(ids)
        assert length >= self.seq_len
        start = np.random.randint(length - self.seq_len)
        sample = ids[start : start + self.seq_len]
        assert len(sample) == self.seq_len
        indexes = np.sort(np.random.choice(self.increase, self.n_choices))
        ids = sample.copy()
        sample[indexes] = MASK_ID
        return sample, ids

    def __len__(self):
        return len(self.lines)


if __name__ == "__main__":
    lst = ENWikiList("enwiki/", 0.1)
    train = lst.to_train_list()
    print("Train size:", len(train), train[23])
    ds = ENWikiDataset(train, 64)
    sample, target = ds[456]
    print(ds.enc.decode(sample))
    print(ds.enc.decode(target))
