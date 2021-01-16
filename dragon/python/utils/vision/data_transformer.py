# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import math
import multiprocessing

import numpy
import PIL.Image
import PIL.ImageEnhance

from dragon.core.framework import config


class DataTransformer(multiprocessing.Process):
    """Decode and transform images."""

    def __init__(self, **kwargs):
        """Create a ``DataTransformer``.

        Parameters
        ----------
        resize : int, optional, default=0
            The size for the shortest edge.
        padding : int, optional, default=0
            The size for zero padding on two sides.
        fill_value : int or sequence, optional, default=127
            The value(s) to fill for padding or cutout.
        crop_size : int, optional, default=0
            The size for random-or-center cropping.
        random_crop_size: int, optional, default=0
            The size for sampling-based random cropping.
        cutout_size : int, optional, default=0
            The square size for the cutout algorithm.
        mirror : bool, optional, default=False
            Whether to apply the mirror (flip horizontally).
        random_scales : Sequence[float], optional, default=(0.08, 1.)
            The range of scales to sample a crop randomly.
        random_aspect_ratios : Sequence[float], optional, default=(0.75, 1.33)
            The range of aspect ratios to sample a crop randomly.
        distort_color : bool, optional, default=False
            Whether to apply color distortion.
        inverse_color : bool, option, default=False
            Whether to inverse channels for color images.
        phase : {'TRAIN', 'TEST'}, optional
            The optional running phase.
        seed : int, optional
            The random seed to use instead.

        """
        super(DataTransformer, self).__init__()
        self._resize = kwargs.get('resize', 0)
        self._padding = kwargs.get('padding', 0)
        self._fill_value = kwargs.get('fill_value', 127)
        self._crop_size = kwargs.get('crop_size', 0)
        self._random_crop_size = kwargs.get('random_crop_size', 0)
        self._cutout_size = kwargs.get('cutout_size', 0)
        self._mirror = kwargs.get('mirror', False)
        self._random_scales = kwargs.get('random_scales', (0.08, 1.))
        self._random_ratios = kwargs.get('random_aspect_ratios', (3. / 4., 4. / 3.))
        self._distort_color = kwargs.get('distort_color', False)
        self._inverse_color = kwargs.get('inverse_color', False)
        self._phase = kwargs.get('phase', 'TRAIN')
        self._seed = kwargs.get('seed', config.config().random_seed)
        self.q_in = self.q_out = None
        self.daemon = True

    def get(self, example):
        """Return image and labels from a serialized str.

        Parameters
        ----------
        example : dict
            The input example.

        Returns
        -------
        numpy.ndarray
            The images.
        Sequence[int]
            The labels.

        """
        # Decode.
        if example['encoded'] > 0:
            img = PIL.Image.open(io.BytesIO(example['data']))
        else:
            img = numpy.frombuffer(example['data'], numpy.uint8)
            img = img.reshape(example['shape'])

        # Resizing.
        if self._resize > 0:
            (w, h), size = img.size, self._resize
            if (w <= h and w == size) or (h <= w and h == size):
                pass
            else:
                if w < h:
                    ow, oh = size, size * h // w
                else:
                    oh, ow = size, size * w // h
                img = img.resize((ow, oh), PIL.Image.BILINEAR)

        # ToArray.
        img = numpy.asarray(img)

        # Padding.
        if self._padding > 0:
            pad_img = numpy.empty((
                img.shape[0] + 2 * self._padding,
                img.shape[1] + 2 * self._padding, img.shape[2]
            ), dtype=img.dtype)
            pad_img[:] = self._fill_value
            pad_img[self._padding:self._padding + img.shape[0],
                    self._padding:self._padding + img.shape[1], :] = img
            img = pad_img

        # Random crop (AlexNet-Style).
        if self._crop_size > 0:
            h = w = self._crop_size
            height, width = img.shape[:2]
            if self._phase == 'TRAIN':
                i = numpy.random.randint(height - h + 1)
                j = numpy.random.randint(width - w + 1)
            else:
                i = (height - h) // 2
                j = (width - w) // 2
            img = img[i:i + h, j:j + w, :]

        # Random crop (Inception-Style).
        if self._random_crop_size > 0:
            height, width = img.shape[:2]
            area = height * width
            i = j = h = w = None
            for attempt in range(10):
                target_area = numpy.random.uniform(*self._random_scales) * area
                log_ratio = (math.log(self._random_ratios[0]),
                             math.log(self._random_ratios[1]))
                aspect_ratio = math.exp(numpy.random.uniform(*log_ratio))
                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))
                if 0 < w <= width and 0 < h <= height:
                    i = numpy.random.randint(height - h + 1)
                    j = numpy.random.randint(width - w + 1)
                    break
            if i is None:
                in_ratio = float(width) / float(height)
                if in_ratio < min(self._random_ratios):
                    w = width
                    h = int(round(w / min(self._random_ratios)))
                elif in_ratio > max(self._random_ratios):
                    h = height
                    w = int(round(h * max(self._random_ratios)))
                else:
                    w, h = width, height
                i = (height - h) // 2
                j = (width - w) // 2
            img = img[i:i + h, j:j + w, :]
            new_size = (self._random_crop_size, self._random_crop_size)
            img = PIL.Image.fromarray(img)
            img = numpy.asarray(img.resize(new_size, PIL.Image.BILINEAR))

        # CutOut.
        if self._cutout_size > 0:
            h, w = img.shape[:2]
            y = numpy.random.randint(h)
            x = numpy.random.randint(w)
            y1 = numpy.clip(y - self._cutout_size // 2, 0, h)
            y2 = numpy.clip(y + self._cutout_size // 2, 0, h)
            x1 = numpy.clip(x - self._cutout_size // 2, 0, w)
            x2 = numpy.clip(x + self._cutout_size // 2, 0, w)
            img[y1:y2, x1:x2] = self._fill_value

        # Random mirror.
        if self._mirror:
            if numpy.random.randint(0, 2) > 0:
                img = img[:, ::-1, :]

        # Color distortion.
        if self._distort_color:
            img = PIL.Image.fromarray(img)
            transforms = [PIL.ImageEnhance.Brightness,
                          PIL.ImageEnhance.Contrast,
                          PIL.ImageEnhance.Color]
            numpy.random.shuffle(transforms)
            for transform in transforms:
                img = transform(img)
                img = img.enhance(1. + numpy.random.uniform(-.4, .4))
            img = numpy.asarray(img)

        # Color transformation.
        if self._inverse_color:
            img = img[:, :, ::-1]

        return img, example['label']

    def run(self):
        """Start the process to produce images."""
        numpy.random.seed(self._seed)

        while True:
            # example -> (image, label)
            self.q_out.put(self.get(self.q_in.get()))
