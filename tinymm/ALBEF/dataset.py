import torch

from torchvision import transforms as v1
from tinymm.CLIP.dataset import CC3MList, CC3MDataset


class ALBEFDataset(CC3MDataset):
    def __init__(self, id_to_filename, lst, seq_len):
        super().__init__(id_to_filename, lst, seq_len)
        self.transforms = v1.Compose(
            [
                v1.RandomCrop((224, 224)),
                # v1.RandAugment(),
            ]
        )

    def __getitem__(self, index):
        image, ids = super().get_raw(index)
        image = torch.tensor(image).permute(2, 0, 1)
        image = self.transforms(image).permute(1, 2, 0).numpy()
        image, ids = super().raw_to_ds(image, ids)
        return image, ids


if __name__ == "__main__":
    lst = CC3MList(("/data/cc3m",), 0.05)
    ds = ALBEFDataset(lst.id_to_filename, lst.to_train_list(), 64)
    length = len(ds)
    for index in range(length):
        image, ids = ds[index]
        assert len(ids) == 64
        print(index, ids)
        print(ds.enc.decode(ids))
        print("\n")
