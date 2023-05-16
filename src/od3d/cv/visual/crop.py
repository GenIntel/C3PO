
import torch
from od3d.cv.visual.resize import resize
def crop(img, center, H_out, W_out, scale=1.):
    device = img.device
    dtype = center.dtype
    img_in_shape = img.shape[1:]
    bbox_in_shape_xhalf = 1. * (W_out / scale) / 2.
    bbox_in_shape_yhalf = 1. * (H_out / scale) / 2.
    #  x0", "y0", "x1", "y1"
    bbox_in = torch.LongTensor([int(center[0] - bbox_in_shape_xhalf),
                                int(center[1] - bbox_in_shape_yhalf),
                                int(center[0] + bbox_in_shape_xhalf),
                                int(center[1] + bbox_in_shape_yhalf)]).to(device)

    # x-, x+, y-, y+
    pad = [max(-bbox_in[0], 0), max(bbox_in[2] - img_in_shape[1], 0), max(-bbox_in[1], 0),
           max(bbox_in[3] - img_in_shape[0], 0)]

    img = torch.nn.functional.pad(img, pad=pad)
    img_crop_bbox_in = img[:, bbox_in[1]+pad[2]: bbox_in[3]+pad[2], bbox_in[0]+pad[0]:bbox_in[2]+pad[0]]

    img_out = resize(img_crop_bbox_in, H_out=H_out, W_out=W_out)

    cam_crop_tform_cam = torch.Tensor([[scale, 0., -bbox_in[0] * scale, 0.],
                                       [0., scale, -bbox_in[1] * scale, 0.],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]]).to(device=device, dtype=dtype)
    return img_out, cam_crop_tform_cam
