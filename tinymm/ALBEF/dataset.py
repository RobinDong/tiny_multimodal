import cv2

from tinymm.CLIP.dataset import CC3MList, CC3MDataset


class ALBEFDataset(CC3MDataset):
    def __getitem__(self, index):
        image, ids = super().get_raw(index)
        image = cv2.resize(image, (224, 224))
        image, ids = super().raw_to_ds(image, ids)
        return image, ids


if __name__ == "__main__":
    lst = CC3MList(("/data/cc3m",), 0.05)
    ds = ALBEFDataset(lst.id_to_filename, lst.to_train_list(), 64)
    length = len(ds)
    for index in range(length):
        image, ids, tgt = ds[index]
        assert len(ids) == 64
        assert len(tgt) == 64
        print(index, ids, tgt)
        print(ds.enc.decode(ids), ds.enc.decode(tgt))
        print("\n")
