![tiny_multimodal](assets/tiny_multimodal.png)

# Tiny Multimodal

A simple and "tiny" implementation of many multimodal models. It supports training/finetuning/deploying these tiny-sized models.
Unlike the popular "large" models, all the models in this repo will be restricted to train on my RTX 3080 Ti so the implementation will not be totally the same to the original papers.

## quick start

### create environment

```
conda create -n tinym python=3.12
conda activate tinym

git clone git@github.com:RobinDong/tiny_multimodal.git
cd tiny_multimodal
python -m pip install -r requirements.txt
```

### prepare dataset for training

Download [conceptual-12m](https://github.com/google-research-datasets/conceptual-12m) from [Huggingface](https://huggingface.co/datasets/pixparse/cc12m-wds) to directory `cc12m-wds`.

Use `python utils/extract_tars.py` to convert CC12M to ready-to-use format.

### train
```
python train.py --provider CLIP
```

## acknowledgements
This repo is still in developing. Please be patient for more multi-modal models.
Any issue or pull request is welcome.
