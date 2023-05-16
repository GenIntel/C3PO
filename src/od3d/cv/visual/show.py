import os
import cv2
from od3d.cv.visual.draw import tensor_to_cv_img
from od3d.cv.visual.resize import resize
def show_img(rgb, duration=0, vwriter=None, fpath=None, height=None, width=None):
    # img: 3xHxW
    rgb = rgb.clone()

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