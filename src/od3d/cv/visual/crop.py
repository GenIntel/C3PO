import logging
logger = logging.getLogger(__name__)
import math
import torch
from od3d.cv.visual.resize import resize
from od3d.cv.visual.show import show_img
def crop(img, center, H_out, W_out, scale=1., ctx=None):
    device = img.device
    dtype = center.dtype
    img_in_shape = img.shape[1:]
    bbox_in_shape_xhalf = 1. * (W_out / scale) / 2.
    bbox_in_shape_yhalf = 1. * (H_out / scale) / 2.

    #  x0", "y0", "x1", "y1"
    bbox_in = torch.LongTensor([math.floor(center[0] - bbox_in_shape_xhalf),
                                math.floor(center[1] - bbox_in_shape_yhalf),
                                math.ceil(center[0] + bbox_in_shape_xhalf),
                                math.ceil(center[1] + bbox_in_shape_yhalf)]).to(device)
    # x-, x+, y-, y+
    pad_in = torch.LongTensor([max(-bbox_in[0], 0), max(bbox_in[2] - img_in_shape[1], 0), max(-bbox_in[1], 0),
              max(bbox_in[3] - img_in_shape[0], 0)]).to(device)



    # two options:
    # a) first crop then resize (preferred if scale > 1. -> pad on lower-resolution image)
    if scale >= 1.:
        # img = torch.nn.functional.pad(img, pad=pad)
        img_padded = torch.zeros(
            size=img.shape[:-2] + torch.Size([img.shape[-2] + pad_in[2] + pad_in[3], img.shape[-1] + pad_in[0] + pad_in[1]]),
            dtype=img.dtype, device=device)
        pad_x_upper = -pad_in[1] if pad_in[1] > 0 else None
        pad_y_upper = -pad_in[3] if pad_in[3] > 0 else None
        img_padded[:, pad_in[2]:pad_y_upper, pad_in[0]:pad_x_upper] = img
        img_cropped = img_padded[:, bbox_in[1] + pad_in[2]: bbox_in[3] + pad_in[2], bbox_in[0] + pad_in[0]:bbox_in[2] + pad_in[0]]
        img_out = resize(img_cropped, H_out=H_out, W_out=W_out)
        logger.info(f'scale < 1. out size: ({img_out.shape[1]}, {img_out.shape[1]})')

    # b) first resize then crop (preferred if scale < 1. -> pad on lower-resolution image)
    else:
        img_res = resize(img, scale_factor=scale)
        bbox_in_res = (bbox_in * scale).to(torch.long)
        bbox_in_res[3] = H_out + bbox_in_res[1]
        bbox_in_res[2] = W_out + bbox_in_res[0]
        pad_in_res = (pad_in * scale).to(torch.long)
        img_padded = torch.zeros(
            size=img_res.shape[:-2] + torch.Size(
                [H_out + pad_in_res[2] + pad_in_res[3], W_out + pad_in_res[0] + pad_in_res[1]]),
            dtype=img_res.dtype, device=device)
        img_padded[:, pad_in_res[2]:pad_in_res[2]+img_res.shape[1], pad_in_res[0]:pad_in_res[0]+img_res.shape[2]] = img_res
        img_out = img_padded[:, bbox_in_res[1] + pad_in_res[2]: bbox_in_res[3] + pad_in_res[2],
                      bbox_in_res[0] + pad_in_res[0]:bbox_in_res[2] + pad_in_res[0]]
        logger.info(f'scale < 1. out size: ({img_out.shape[1]}, {img_out.shape[1]})')

    if ctx is not None:
        bbox_out = torch.LongTensor([math.ceil(pad_in[0] * scale),
                                     math.ceil(pad_in[2] * scale),
                                     math.floor(W_out - 1 - pad_in[1] * scale),
                                     math.floor(H_out - 1 - pad_in[3] * scale)]).to(device)
        ctx = resize(ctx, H_out=H_out, W_out=W_out)
        ctx[:, bbox_out[1]:bbox_out[3], bbox_out[0]:bbox_out[2]] = img_out[:, bbox_out[1]: bbox_out[3], bbox_out[0]:bbox_out[2]]
        img_out = ctx

    cam_crop_tform_cam = torch.Tensor([[scale, 0., -bbox_in[0] * scale, 0.],
                                       [0., scale, -bbox_in[1] * scale, 0.],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]]).to(device=device, dtype=dtype)
    return img_out, cam_crop_tform_cam
