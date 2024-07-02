import torch

from torchvision import transforms as v1
from tinymm.CLIP.dataset import CC3MList, CC3MDataset


def aug_space_without_color(num_bins: int, image_size: tuple[int, int]):
    return {
        # op_name: (magnitudes, signed)
        "Identity": (torch.tensor(0.0), False),
        "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
        "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
        "TranslateX": (
            torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins),
            True,
        ),
        "TranslateY": (
            torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins),
            True,
        ),
        "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
        "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
        # "Color": (torch.linspace(0.0, 0.9, num_bins), True),
        "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
        "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
        "Posterize": (
            8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
            False,
        ),
        "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (torch.tensor(0.0), False),
        "Equalize": (torch.tensor(0.0), False),
    }


class ALBEFDataset(CC3MDataset):
    def __init__(self, id_to_filename, lst, seq_len):
        super().__init__(id_to_filename, lst, seq_len)
        ra = v1.RandAugment()
        ra._augmentation_space = aug_space_without_color
        self.transforms = v1.Compose(
            [
                v1.RandomCrop((224, 224)),
                ra,
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
