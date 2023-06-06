from pathlib import Path
from PIL import Image
from torchvision import transforms
from pytorch3d.io import load_ply
from pytorch3d.io import save_ply
import numpy as np
import torch

def read_co3d_depth_image(path: Path):
    img = Image.open(path)

    img = (
        np.frombuffer(np.array(img, dtype=np.uint16), dtype=np.float16)
        .astype(np.float32)
        .reshape((img.size[1], img.size[0]))
    )
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img = transform(img)
    return img

def save_image_mask(img: torch.Tensor, path: Path):
    transform = transforms.Compose([
        transforms.ToPILImage()
    ])
    img = transform(img.to(torch.uint8) * 255)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def read_image(path: Path):
    img = Image.open(path)

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # Convert the PIL image to Torch tensor
    img = transform(img)
    return img
