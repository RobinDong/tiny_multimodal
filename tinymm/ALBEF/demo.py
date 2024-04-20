import cv2
import fire

import torch
import numpy as np
import streamlit as st

from transformers import BertTokenizerFast
from tinymm.utils import load_from_checkpoint

MASK_ID = 50258


class Demo:
    image_path: str = "/home/robin/Downloads/imagenet/val"
    image_size: tuple = (256, 256)
    seq_len: int = 64

    def __init__(self):
        self.images = {}
        self.enc = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

    def load_model(self, checkpoint: str):
        model = load_from_checkpoint(checkpoint)
        return model

    def start(self, checkpoint: str, cuda: bool = False):
        model = self.load_model(checkpoint)
        model.eval()
        uploaded_file = st.file_uploader(
            "Choose a image file", type=["jpg", "png", "webp"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.resize(image, self.image_size)
            st.image(image, channels="BGR")

        text = st.text_input("Input:")
        if not text:
            return
        ids = self.enc(text)["input_ids"]
        print("ids:", ids)
        ids = np.pad(ids, (0, (self.seq_len - len(ids))), "constant")
        ids = torch.tensor(ids).unsqueeze(0)

        image = image.astype("float32") / 255.0
        images = torch.tensor(image).unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
        with torch.no_grad():
            _, _, _, logits, _, _, _, _ = model((images, ids, ids))
        print("logits:", logits.size(), logits)
        _, predict = torch.max(logits, dim=-1)
        print("predict:", predict.size(), predict)
        decoded = self.enc.decode(predict[0].tolist())
        st.text(decoded)
        print("decoded:", decoded)


if __name__ == "__main__":
    demo = Demo()
    fire.Fire(demo.start)
