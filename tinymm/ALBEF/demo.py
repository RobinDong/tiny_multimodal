import cv2
import fire

import torch
import numpy as np
import streamlit as st

from transformers import BertTokenizerFast
from tinymm.utils import load_from_checkpoint


class Demo:
    image_path: str = "/home/robin/Downloads/imagenet/val"
    image_size: tuple = (224, 224)
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
        if not uploaded_file:
            return

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
            logits = model((images, ids), True)
        print("logits:", logits.size(), logits)
        _, predict = torch.max(logits, dim=-1)
        print("predict:", predict.size(), predict)
        decoded = self.enc.decode(predict[0].tolist())
        ans = decoded.split(",")[0]
        st.text(ans)
        print("ans:", ans)


if __name__ == "__main__":
    demo = Demo()
    fire.Fire(demo.start)
