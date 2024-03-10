import cv2
import glob
import tarfile

import numpy as np

from multiprocessing import Pool

n_jobs = 12
input_path = "/mnt/backup_3080ti/cc12m/"
output_path = "/home/robin/Downloads/cc12m/"
image_size = (256, 256)


def task(tar_filename):
    drop_list = set()
    try:
        tarf = tarfile.open(tar_filename)
        for member in tarf.getmembers():
            if member.name.endswith(".txt"):
                txt_fp = tarf.extractfile(member)
                content = txt_fp.read()
                try:
                    text = content.decode("ascii")
                except Exception:
                    drop_list.add(member.name[:-4])
                    continue
                with open(output_path + member.name, "w") as fp:
                    fp.write(text)
            if member.name.endswith(".jpg"):
                if member.name[:-4] in drop_list:
                    continue
                image_fp = tarf.extractfile(member)
                image_content = np.asarray(bytearray(image_fp.read()), dtype="uint8")
                image = cv2.imdecode(image_content, cv2.IMREAD_COLOR)
                image = cv2.resize(image, image_size)
                cv2.imwrite(output_path + member.name, image)
    except Exception as ex:
        print("Exception:", ex)
        print("Failed filename:", tar_filename)


def extract():
    tar_list = glob.glob(input_path + "*.tar")
    with Pool(n_jobs) as pool:
        pool.map(task, tar_list)


if __name__ == "__main__":
    extract()
