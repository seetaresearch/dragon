# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import multiprocessing

from dragon import config as _cfg
from dragon.vm.caffe.proto import caffe_pb2 as _proto_def

try:
    import cv2
except ImportError as e:
    print('Failed to import cv2. Error: {0}'.format(str(e)))
try:
    import PIL.Image
    import PIL.ImageEnhance
except ImportError as e:
    print("Failed to import PIL. \nIt's OK if disabling color augmentation.".format(str(e)))


class DataTransformer(multiprocessing.Process):
    """DataTransformer is deployed to queue transformed images from `DataReader`_.

    Nearly all common image augmentation methods are supported.

    """
    def __init__(self, **kwargs):
        """Construct a ``DataTransformer``.

        Parameters
        ----------
        padding : int, optional, default=0
            The zero-padding size.
        fill_value : int or sequence, optional, default=127
            The value(s) to fill for padding or cutout.
        crop_size : int, optional, default=0
            The cropping size.
        cutout_size : int, optional, default=0
            The square size to cutout.
        mirror : bool, optional, default=False
            Whether to mirror(flip horizontally) images.
        color_augmentation : bool, optional, default=False
            Whether to use color distortion.1
        min_random_scale : float, optional, default=1.
            The min scale of the input images.
        max_random_scale : float, optional, default=1.
            The max scale of the input images.
        force_gray : bool, optional, default=False
            Set not to duplicate channel for gray.
        phase : {'TRAIN', 'TEST'}, optional
            The optional running phase.

        """
        super(DataTransformer, self).__init__()
        self._padding = kwargs.get('padding', 0)
        self._fill_value = kwargs.get('fill_value', 127)
        self._crop_size = kwargs.get('crop_size', 0)
        self._cutout_size = kwargs.get('cutout_size', 0)
        self._mirror = kwargs.get('mirror', False)
        self._color_aug = kwargs.get('color_augmentation', False)
        self._min_rand_scale = kwargs.get('min_random_scale', 1.0)
        self._max_rand_scale = kwargs.get('max_random_scale', 1.0)
        self._force_color = kwargs.get('force_color', False)
        self._phase = kwargs.get('phase', 'TRAIN')
        self._rng_seed = _cfg.GetRandomSeed()
        self.Q_in = self.Q_out = None
        self.daemon = True

    def get(self, serialized):
        """Return image and labels from a serialized str.

        Parameters
        ----------
        serialized : str
            The protobuf serialized str.

        Returns
        -------
        tuple
            The tuple image and labels.

        """
        # Decode
        datum = _proto_def.Datum()
        datum.ParseFromString(serialized)
        im = numpy.fromstring(datum.data, numpy.uint8)
        if datum.encoded is True:
            im = cv2.imdecode(im, -1)
        else:
            im = im.reshape((datum.height, datum.width, datum.channels))

        # Random scale
        rand_scale = numpy.random.uniform() * (
            self._max_rand_scale - self._min_rand_scale
                ) + self._min_rand_scale
        if rand_scale != 1.0:
            im = cv2.resize(
                im, None,
                fx=rand_scale,
                fy=rand_scale,
                interpolation=cv2.INTER_LINEAR,
            )

        # Padding
        if self._padding > 0:
            pad_im = numpy.empty((
                im.shape[0] + 2 * self._padding,
                im.shape[1] + 2 * self._padding, im.shape[2]
            ), dtype=im.dtype)
            pad_im[:] = self._fill_value
            pad_im[self._padding : self._padding + im.shape[0],
                   self._padding : self._padding + im.shape[1], :] = im
            im = pad_im

        # Random crop
        if self._crop_size > 0:
            if self._phase == 'TRAIN':
                h_off = numpy.random.randint(im.shape[0] - self._crop_size + 1)
                w_off = numpy.random.randint(im.shape[1] - self._crop_size + 1)
            else:
                h_off = int((im.shape[0] - self._crop_size) / 2)
                w_off = int((im.shape[1] - self._crop_size) / 2)
            im = im[h_off : h_off + self._crop_size,
                    w_off : w_off + self._crop_size, :]

        # CutOut
        if self._cutout_size > 0:
            h, w = im.shape[:2]
            y = numpy.random.randint(h)
            x = numpy.random.randint(w)
            y1 = numpy.clip(y - self._cutout_size // 2, 0, h)
            y2 = numpy.clip(y + self._cutout_size // 2, 0, h)
            x1 = numpy.clip(x - self._cutout_size // 2, 0, w)
            x2 = numpy.clip(x + self._cutout_size // 2, 0, w)
            im[y1 : y2, x1 : x2] = self._fill_value

        # Random mirror
        if self._mirror:
            if numpy.random.randint(0, 2) > 0:
                im = im[:, ::-1, :]

        # Gray Transformation
        if self._force_color:
            if im.shape[2] == 1:
                # Duplicate to 3 channels
                im = numpy.concatenate([im, im, im], axis=2)

        # Color Augmentation
        if self._color_aug:
            im = PIL.Image.fromarray(im)
            delta_brightness = numpy.random.uniform(-0.4, 0.4) + 1.0
            delta_contrast = numpy.random.uniform(-0.4, 0.4) + 1.0
            delta_saturation = numpy.random.uniform(-0.4, 0.4) + 1.0
            im = PIL.ImageEnhance.Brightness(im)
            im = im.enhance(delta_brightness)
            im = PIL.ImageEnhance.Contrast(im)
            im = im.enhance(delta_contrast)
            im = PIL.ImageEnhance.Color(im)
            im = im.enhance(delta_saturation)
            im = numpy.array(im)

        # Extract Labels
        labels = []
        if len(datum.labels) > 0: labels.extend(datum.labels)
        else: labels.append(datum.label)

        return im, labels

    def run(self):
        """Start the process.

        Returns
        -------
        None

        """
        # Fix the random seed
        numpy.random.seed(self._rng_seed)

        # Run!
        while True:
            serialized = self.Q_in.get()
            self.Q_out.put(self.get(serialized))