import cv2
import glob
import tiktoken

import numpy as np
import torch.utils.data as data


class CC3MList:
    def __init__(self, path, eval_ratio):
        pair_list = []
        # list all jpg/txt files
        lst = glob.glob(f"{path}/*.txt")
        for txt_name in lst:
            if txt_name.endswith(
                "000007279.txt"
            ):  # The "000007279.txt" contains float numbers, which is not suitable
                continue
            file_name = txt_name[:-4]
            pair_list.append((file_name + ".jpg", txt_name))
        self.pair_list = pair_list
        self.div = int(len(pair_list) * (1 - eval_ratio))

    def to_train_list(self):
        return self.pair_list[: self.div]

    def to_eval_list(self):
        return self.pair_list[self.div :]


class CC3MDataset(data.Dataset):
    def __init__(self, lst, seq_len):
        self.pair_list = lst
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")

    def __getitem__(self, index):
        image_name, txt_name = self.pair_list[index]

        image = cv2.imread(image_name)
        with open(txt_name, "r") as fp:
            txt = fp.read()
        ids = self.enc.encode_ordinary(txt)
        length = len(ids)
        if length > self.seq_len:
            ids = ids[: self.seq_len]
            print("ids:", ids, txt_name)
        elif length < self.seq_len:
            ids = np.pad(ids, (0, (self.seq_len - length)), "constant")
        imgs = image.astype("float32") / 255.0
        return imgs, ids

    def __len__(self):
        return len(self.pair_list)


if __name__ == "__main__":
    lst = CC3MList("/home/robin/Downloads/CC3M", 0.1)
    ds = CC3MDataset(lst.to_eval_list())
    image, txt = ds[123]
    print(image)
    print(txt)
