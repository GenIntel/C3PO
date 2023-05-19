import torch

def blend_rgb(rgb1, rgb2, alpha1=0.5, alpha2=0.5):
    if rgb1.dtype == torch.bool:
        rgb1 = rgb1 * 255.
    if rgb2.dtype == torch.bool:
        rgb2 = rgb2 * 255.
    return (alpha1 * rgb1 + alpha2 * rgb2).to(torch.uint8)