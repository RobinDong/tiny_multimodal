import glob
import cv2
import tiktoken

import numpy as np
from torch.utils import data


class CC3MList:
    def __init__(self, pathes, eval_ratio):
        # list all index(.npy) files
        lst = []
        for path in pathes:
            lst += glob.glob(f"{path}/*.npy")
        np.random.shuffle(lst)
        filename_to_id = {}
        self._id_to_filename = {}
        fid = 0
        self.indexes = []
        for index_file in lst:
            index = np.load(index_file)
            filename = index_file[:-4]
            if filename not in filename_to_id:
                file_id = fid
                filename_to_id[filename] = file_id
                self._id_to_filename[file_id] = filename
                fid += 1
            else:
                file_id = filename_to_id[filename]
            for item in index:
                self.indexes.append((item, file_id))
        print("Dataset size:", len(self.indexes))
        self.div = int(len(self.indexes) * (1 - eval_ratio))

    def to_train_list(self):
        return self.indexes[: self.div]

    def to_eval_list(self):
        return self.indexes[self.div :]

    @property
    def id_to_filename(self):
        return self._id_to_filename


class CC3MDataset(data.Dataset):
    def __init__(self, id_to_filename, lst, seq_len):
        self.id_to_filename = id_to_filename
        self.indexes = lst
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")

    def __getitem__(self, index):
        img_offset, img_size, txt_offset, txt_size = self.indexes[index][0]
        filename = self.id_to_filename[self.indexes[index][1]]
        with open(filename + ".dat", "rb") as fp:
            fp.seek(img_offset)
            raw_image = np.asarray(bytearray(fp.read(img_size)), dtype="uint8")
            image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
            fp.seek(txt_offset)
            text = fp.read(txt_size).decode("utf-8")

        ids = self.enc.encode_ordinary(text)
        ids = np.array(ids, dtype=np.int64)
        length = len(ids)
        if length > self.seq_len:
            ids = ids[: self.seq_len]
        elif length < self.seq_len:
            ids = np.pad(ids, (0, (self.seq_len - length)), "constant")
        imgs = image.astype("float32") / 255.0
        return imgs, ids

    def __len__(self):
        return len(self.indexes)


if __name__ == "__main__":
    lst = CC3MList("/home/robin/Downloads/cc12m", 0.1)
    ds = CC3MDataset(lst.id_to_filename, lst.to_eval_list(), 64)
    length = len(ds)
    for index in range(length):
        image, txt = ds[index]
        print(txt.shape, txt)
