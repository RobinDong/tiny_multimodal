import torch
import fire

from tinymm.utils import load_from_checkpoint


def dynamic_quantize(checkpoint: str, output: str = "quantized.qpt"):
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = load_from_checkpoint(checkpoint)
    qmodel = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    ckpt["quantization"] = "dynamic"
    ckpt["model"] = qmodel
    torch.save(ckpt, output)


if __name__ == "__main__":
    fire.Fire(dynamic_quantize)
