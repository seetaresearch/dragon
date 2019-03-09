# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
#
# Codes are based on:
#
#      <https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import cv2
from PIL import Image, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import collections
import warnings


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3, 4})


def to_array(pic, normalize=True):
    """Convert a ``PIL Image`` to ``numpy.ndarray``

    """
    if not _is_numpy_image(pic):
        raise TypeError('pic should be numpy Image. Got {}'.format(type(pic)))
    if normalize: return pic.astype(np.float32) / 255
    else: return pic.astype(np.float32)


def normalize(array, mean, std):
    if not _is_numpy_image(array):
        raise TypeError('array is not a numpy image.')
    array = (array - mean) / std
    array = array.astype(np.float32)
    if len(array.shape) == 3: return array.transpose(2, 0, 1)
    elif len(array.shape) == 4: return array.transpose(0, 3, 1, 2)
    else: raise ValueError('excepted a 3d or 4d array.')


_pil_interpolation_to_ocv = {
    Image.NEAREST: cv2.INTER_NEAREST,
    Image.BILINEAR: cv2.INTER_LINEAR,
    Image.BICUBIC: cv2.INTER_CUBIC,
    Image.LANCZOS: cv2.INTER_LANCZOS4
}


def resize(img, size, interpolation=Image.BILINEAR):
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w = img.shape[0:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else: oh, ow = size
    if sys.version_info >= (3, 0):
        return cv2.resize(img, dsize=(ow, oh),
            interpolation=_pil_interpolation_to_ocv[interpolation])
    else:
        # Fuck Fuck Fuck opencv-python2, it always has a BUG
        # that leads to duplicate cuDA handles created at gpu:0
        img = Image.fromarray(img)
        img = img.resize((ow, oh), interpolation)
        return np.array(img)


def scale(*args, **kwargs):
    warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                  "please use transforms.Resize instead.")
    return resize(*args, **kwargs)


def pad(img, padding, fill=0, padding_mode='constant'):
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    # RGB image
    if len(img.shape) == 3:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
    # Grayscale image
    if len(img.shape) == 2:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

    return img


def crop(img, i, j, h, w, copy=False):
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    if copy: return img[i : i + h, j : j + w].copy()
    return img[i : i + h, j : j + w]


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = img.shape[0:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    assert _is_numpy_image(img), 'img should be numpy Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


def hflip(img):
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    return img[:, ::-1]


def vflip(img):
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    return img[::-1, :]


def five_crop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    h, w = img.shape[0:2]
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size, (h, w)))
    tl = crop(img, 0, 0, crop_h, crop_w, True)
    tr = crop(img, 0, w - crop_w, crop_h, crop_w, True)
    bl = crop(img, h - crop_h, 0, crop_h, crop_w, True)
    br = crop(img, h - crop_h, w - crop_w, crop_h, crop_w, True)
    center = center_crop(img, (crop_h, crop_w)).copy()
    return np.stack((tl, tr, bl, br, center), axis=0)


def ten_crop(img, size, vertical_flip=False):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return np.concatenate([first_five, second_five], axis=0)


def adjust_brightness(img, brightness_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img