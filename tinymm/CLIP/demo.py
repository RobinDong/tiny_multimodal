import os
import fire
import glob
import pickle

import torch
import numpy as np
import streamlit as st

from tqdm import tqdm
from transformers import BertTokenizerFast
from tinymm.utils import load_from_checkpoint, load_image


class Demo:
    image_paths: list[str] = ["/data/imagenet/val", "/data/imagenet/test"]
    image_size: tuple = (256, 256)
    seq_len: int = 64
    embd_path: str = "embd.pkl"

    def __init__(self):
        self.images = {}
        self.enc = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

    def load_model(self, checkpoint: str):
        model = load_from_checkpoint(checkpoint)
        return model

    def build_embeddings(self, model, cuda):
        if os.path.exists(self.embd_path):
            with open(self.embd_path, "rb") as fp:
                self.images = pickle.load(fp)
            return

        img_lst = []
        for path in self.image_paths:
            img_lst += glob.glob(f"{path}/*.JPEG")
        print(f"Number of images: {len(img_lst)}")
        for image_name in tqdm(img_lst):
            image = load_image(image_name, self.image_size)
            model.img_encoder = model.img_encoder
            if cuda:
                image = image.cuda()
                model.img_encoder = model.img_encoder.cuda()
            with torch.no_grad():
                image_embd = model.img_encoder(image)
                if cuda:
                    image_embd = image_embd.cpu()
            self.images[image_name] = image_embd

        with open(self.embd_path, "wb") as fp:
            pickle.dump(self.images, fp)

    def start(self, checkpoint: str, cuda: bool = False):
        model = self.load_model(checkpoint)
        model.eval()
        self.build_embeddings(model, cuda)

        text = st.text_input("Input:")
        if not text:
            return
        ids = self.enc(text)["input_ids"]
        ids = np.pad(ids, (0, (self.seq_len - len(ids))), "constant")
        ids = torch.tensor(ids).unsqueeze(0)
        txt_embd = model.txt_encoder(ids)

        result = []
        for image_name, image_embd in self.images.items():
            result.append((image_name, image_embd @ txt_embd.T))
        result = sorted(result, key=lambda pair: pair[1], reverse=True)
        for index in range(10):
            st.image(result[index][0])


if __name__ == "__main__":
    demo = Demo()
    fire.Fire(demo.start)
