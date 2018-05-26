import os
import shutil
import numpy as np
import cv2
from shapes import Rectangle, Shape


# checked
def fill_zero(x: np.array, rect: Rectangle):
    assert isinstance(rect, (Shape, list, tuple, set))
    if isinstance(rect, Shape):
        upper, lower, left, right = rect.get_upper(), rect.get_lower(), rect.get_left(), rect.get_right()
        if len(x.shape) == 3:
            x[upper:lower, left:right, :] = 0
        else:
            x[upper:lower, left:right] = 0
    else:
        for r in rect:
            fill_zero(x, r)


def find_bounding_box(xx, yy, pix_count_thres=2000):
    try:
        assert len(xx) >= pix_count_thres
        upper = int(np.percentile(xx, 25))
        lower = int(np.percentile(xx, 75))
        m = int(np.percentile(xx, 5))
        M = int(np.percentile(xx, 95))
        IQR = lower - upper
        upper, lower = int(upper - 0.8 * IQR), int(lower + 0.8 * IQR)
        upper, lower = np.maximum(m, upper), np.minimum(M, lower)

        left = int(np.percentile(yy, 25))
        right = int(np.percentile(yy, 75))
        IQR = right - left
        m = int(np.percentile(yy, 5))

        M = int(np.percentile(yy, 95))
        left, right = int(left - 0.8 * IQR), int(right + 0.8 * IQR)
        left, right = np.maximum(m, left), np.minimum(M, right)
    except:
        upper = lower = left = right = 0

    return upper, lower, left, right


def coords_within_boundary(xx, yy, upper, lower, left, right, zero_mean=False):
    # print("inside coords_within_boundary()")
    # print(upper, lower, left, right)
    coords = np.stack((xx, yy), 1)
    # print(coords.shape)
    cond1 = np.logical_and(xx >= upper, xx <= lower)
    cond2 = np.logical_and(yy >= left, yy <= right)
    cond = np.logical_and(cond1, cond2)
    cond = np.stack([cond, cond], 1)
    # print(cond.shape)
    # print(coords[cond])
    coords = np.reshape(coords[cond], (-1, 2))
    # print(coords.shape)
    if zero_mean:
        coords = coords - np.mean(coords, 0)
        # print(coords.shape)
    return coords


def force_mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def get_foreground(frame: np.array, background: np.array, pix_diff_thres):
    foreground = np.abs(frame.astype(np.int32) - background.astype(np.int32)).astype(np.uint8)
    foreground[foreground < pix_diff_thres] = 0
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    return foreground
