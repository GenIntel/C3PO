import logging
logger = logging.getLogger(__name__)
import os
import cv2
from od3d.cv.visual.draw import tensor_to_cv_img
from od3d.cv.visual.resize import resize
import torch
import math
def show_imgs(rgbs, duration=0, vwriter=None, fpath=None, height=None, width=None):
    # rgb: K x 3 x H x W / GH x GW x 3 x H x W

    # , masks_mulitply=None, masks_overlay=None
    #rgbs = (rgbs * 1.0).clamp(0, 1)
    #if masks is not None:
    #    rgb = (rgbs + masks) / 2.0

    rgbs = torch.nn.functional.pad(rgbs, (1, 1, 1, 1), "constant", 1.0)
    #margin = 2
    #torch.nn.functional.pad(rgbs, (1, 1), "constant", 0)

    rgbs_shape = rgbs.shape

    if len(rgbs_shape) == 5:
        GH, GW, C, H, W = rgbs_shape
        rgb = rgbs.clone()
    elif len(rgbs_shape) == 4:
        K, C, H, W = rgbs.shape
        prop_w = 4
        prop_h = 3
        GW = math.ceil(math.sqrt((K * prop_w ** 2) / prop_h ** 2))
        GH = math.ceil(K / GW)
        GTOTAL = GH * GW

        img_placeholder = torch.zeros_like(rgbs[:1]).repeat(GTOTAL - K, 1, 1, 1)

        rgb = torch.cat((rgbs, img_placeholder), dim=0)

        rgb = rgb.reshape(GH, GW, C, H, W)
    else:
        logger.error('Visualize imgs requires the input rgb tensor to have 4 (KxCxHxW) or 5 (GHxGWxCxHxW) dimensions')
        raise NotImplementedError

    rgb = rgb.permute(2, 0, 3, 1, 4)

    rgb = rgb.reshape(C, GH * H, GW * W)

    show_img(rgb, duration, vwriter, fpath, height, width)

def show_img(rgb, duration=0, vwriter=None, fpath=None, height=None, width=None, normalize=False):
    # img: 3xHxW
    rgb = rgb.clone()

    if normalize:
        rgb = (rgb -rgb.min()) / (rgb.max() - rgb.min())

    if width is not None:
        orig_width = rgb.size(2)
        scale_factor = width / orig_width

    elif height is not None:
        orig_height = rgb.size(1)
        scale_factor = height / orig_height

    if width or height is not None:
        rgb = resize(
            rgb[
                None,
            ],
            scale_factor=scale_factor,
        )[0]

    img = tensor_to_cv_img(rgb)

    if vwriter is not None:
        vwriter.write(img)

    if fpath is not None:
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        cv2.imwrite(fpath, img)
    else:
        cv2.imshow("img", img)
        cv2.waitKey(duration)