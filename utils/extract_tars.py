import fire
import glob
import tarfile
from functools import partial
from multiprocessing import Pool
from collections import defaultdict

import cv2
import numpy as np


def task(tar_filename, output_path, image_size):
    mapping = defaultdict(
        dict
    )  # filename -> {"img" -> image-bytes, "txt" -> text-bytes}
    try:
        with tarfile.open(tar_filename) as tarf:
            for member in tarf.getmembers():
                if member.name.endswith(".txt"):
                    txt_fp = tarf.extractfile(member)
                    content = txt_fp.read()
                    try:
                        text = content.decode("ascii")
                    except UnicodeDecodeError:
                        continue
                    mapping[member.name[:-4]]["txt"] = text.encode("utf-8")
                if member.name.endswith(".jpg"):
                    image_fp = tarf.extractfile(member)
                    image_content = np.asarray(
                        bytearray(image_fp.read()), dtype="uint8"
                    )
                    image = cv2.imdecode(image_content, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, image_size)
                    success, img_bytes = cv2.imencode(".jpg", image)
                    if not success:
                        continue
                    mapping[member.name[:-4]]["img"] = img_bytes
        file_index = []
        offset = 0
        new_filename = output_path + tar_filename.split("/")[-1][:-4]
        data_filename = new_filename + ".dat"
        index_filename = new_filename + ".npy"
        with open(data_filename, "wb") as fp:
            for value in mapping.values():
                if len(value) != 2:
                    continue
                imgb, txtb = value["img"], value["txt"]
                fp.write(imgb)
                fp.write(txtb)
                file_index.append([offset, len(imgb), offset + len(imgb), len(txtb)])
                offset += len(imgb) + len(txtb)
        with open(index_filename, "wb") as fp:
            np.save(fp, np.asarray(file_index))

    except OSError as ex:
        print("Exception:", ex)
        print("Failed filename:", tar_filename)


def extract(
    input_path="/mnt/backup_3080ti/cc12m/",
    output_path="/home/robin/Downloads/cc12m/",
    jobs=2,
    image_size=(256, 256),
):
    tar_list = glob.glob(input_path + "*.tar")
    with Pool(jobs) as pool:
        pool.map(
            partial(task, output_path=output_path, image_size=image_size), tar_list
        )


if __name__ == "__main__":
    fire.Fire(extract)
