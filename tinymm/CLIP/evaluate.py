import glob
import fire
import tiktoken
import numpy as np

import torch

from tqdm import tqdm
from tinymm.utils import load_from_checkpoint, load_image


class Imagenet1K:
    image_path: str = "/home/robin/Downloads/imagenet/val"
    map_file_path: str = "/home/robin/Downloads/imagenet_map.txt"
    nr_categories: int = 1000
    seq_len: int = 64
    image_size: tuple = (256, 256)

    def __init__(self, model):
        self.id_to_index = {}
        self.cats = {}
        with open(self.map_file_path, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                _id, index, name = line.split(" ")
                index = int(index)
                self.cats[index] = name.replace("_", " ")
                self.id_to_index[_id] = index - 1

        self.images = {}
        img_lst = glob.glob(f"{self.image_path}/*.JPEG")
        for image_name in img_lst:
            _id = image_name.split(".")[0].split("_")[-1]
            self.images[image_name] = self.id_to_index[_id]

        self.model = model
        # embedings of all categories
        enc = tiktoken.get_encoding("gpt2")
        cat_embds = []
        print("Compute embeddings for all categories...")
        for index in tqdm(range(1, self.nr_categories + 1)):
            text = f"A photo of a {self.cats[index]}"
            ids = enc.encode_ordinary(text)
            ids = np.pad(ids, (0, (self.seq_len - len(ids))), "constant")
            ids = torch.tensor(ids).unsqueeze(0)
            with torch.no_grad():
                cat_embds.append(self.model.txt_encoder(ids))
        self.cat_embds = torch.cat(cat_embds, dim=0)

    def evaluate(self):
        correct = 0
        processed = 0
        idx = 0
        print("Check all images...")
        for image_name, correct_index in self.images.items():
            image = load_image(image_name)
            with torch.no_grad():
                image_embd = self.model.img_encoder(image)
            logits_per_image = image_embd @ self.cat_embds.T
            # _, _max = torch.max(logits_per_image, dim=-1)
            # if _max.item() == correct_index:
            _, indices = torch.topk(logits_per_image, 5, dim=-1)
            top5 = set(indices.tolist()[0])
            if correct_index in top5:
                correct += 1
            processed += 1
            idx += 1
            if idx % 1000 == 0:
                print(f"Accuracy: {correct*100/processed:.4f}%")


class Evaluator:
    class_map: dict = {
        "imagenet1k": Imagenet1K,
    }

    def evaluate(self, checkpoint: str, dataset="imagenet1k"):
        model = load_from_checkpoint(checkpoint)
        dataset = self.class_map[dataset](model)
        dataset.evaluate()


if __name__ == "__main__":
    eva = Evaluator()
    fire.Fire(eva.evaluate)
