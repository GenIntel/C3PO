import torch

def sample_pxl2d_pts(x, pxl2d, padding_mode='zeros'):
    """
    Args:
        x (torch.Tensor): CxHxW / BxCxHxW
        pxl2d (torch.Tensor): Nx2 / BxNx2

    Returns:
        x_sampled (torch.Tensor): NxC / BxNxC
    """

    dtype=x.dtype
    if dtype == torch.uint8 or dtype == torch.bool:
        x = x.to(dtype=pxl2d.dtype)

    if x.dim() == 3:
        x_batched = False
        x = x[None, ]
    else:
        x_batched = True

    if pxl2d.dim() == 2:
        pxl2d_batched = False
        pxl2d = pxl2d[None,]
    else:
        pxl2d_batched = True

    H, W = x.shape[-2:]

    pxl2d_normalized = pxl2d.clone()
    pxl2d_normalized[:, :, 0] = pxl2d[:, :, 0] / (W - 1.0) * 2.0 - 1.0
    pxl2d_normalized[:, :, 1] = pxl2d[:, :, 1] / (H - 1.0) * 2.0 - 1.0

    # input: (B, C, Hin​, Win​) and grid: (1, Hout​, Wout​, 2)
    # output: (B, C, Hin, Win
    # TODO: delete float(), which is required for 16-bit precision (if pytorch version > 1.7.1 it might be fixed)
    if padding_mode == 'ones':
        _padding_mode = 'zeros'
    else:
        _padding_mode = padding_mode
    x_sampled = torch.nn.functional.grid_sample(
        input=x,
        grid=pxl2d_normalized[:, None],
        mode='bilinear',
        padding_mode=_padding_mode,
        align_corners=True,
    )[:, :, 0]
    x_sampled = x_sampled.permute(0, 2, 1)

    if padding_mode == 'ones':
        mask_outside_of_grid = (pxl2d_normalized[:, :, 0] < -1.) + (pxl2d_normalized[:, :, 1] < -1.) + (pxl2d_normalized[:, :, 0] > 1.) + (pxl2d_normalized[:, :, 1] > 1.)
        x_sampled[mask_outside_of_grid] = 1.

    if not pxl2d_batched and not x_batched:
        x_sampled = x_sampled[0]

    x_sampled = x_sampled.to(dtype=dtype)
    return x_sampled

def sample_pxl2d_grid(img, pxl2d):
    """
    Args:
        img (torch.Tensor): CxHxW / BxCxHxW
        pxl2d (torch.Tensor): 2xH'xW' / Bx2xH'xW'

    Returns:
        img_sampled (torch.Tensor): CxH'xW' / BxCxH'xW'
    """
    pass