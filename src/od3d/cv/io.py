from pathlib import Path
from PIL import Image
from torchvision import transforms
from pytorch3d.io import load_ply


def read_image(path: Path):
    img = Image.open(path)

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # Convert the PIL image to Torch tensor
    img = transform(img)
    return img