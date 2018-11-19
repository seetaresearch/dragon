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

import sys
import numpy as np
import numpy.random as npr
from multiprocessing import Process

import dragon.config as config
import dragon.vm.caffe.proto.caffe_pb2 as pb

try:
    import cv2
except ImportError as e:
    print('Failed to import cv2. Error: {0}'.format(str(e)))
try:
    import PIL.Image
    import PIL.ImageEnhance
except ImportError as e:
    print("Failed to import PIL. \nIt's OK if disabling color augmentation.".format(str(e)))


class DataTransformer(Process):
    """DataTransformer is deployed to queue transformed images from `DataReader`_.

    Nearly all common image augmentation methods are supported.

    """
    def __init__(self, **kwargs):
        """Construct a ``DataTransformer``.

        Parameters
        ----------
        padding : int
            The padding size. Default is ``0`` (Disabled).
        fill_value : int
            The value to fill when padding is valid. Default is ``127``.
        crop_size : int
            The crop size. Default is ``0`` (Disabled).
        mirror : boolean
            Whether to flip(horizontally) images. Default is ``False``.
        color_augmentation : boolean
            Whether to distort colors. Default is ``False``.
        min_random_scale : float
            The min scale of the input images. Default is ``1.0``.
        max_random_scale : float
            The max scale of the input images. Default is ``1.0``.
        force_color : boolean
            Set to duplicate channels for gray. Default is ``False``.
        phase : str
            The phase of this operator, ``TRAIN`` or ``TEST``. Default is ``TRAIN``.

        """
        super(DataTransformer, self).__init__()
        self._padding = kwargs.get('padding', 0)
        self._fill_value = kwargs.get('fill_value', 127)
        self._crop_size = kwargs.get('crop_size', 0)
        self._mirror = kwargs.get('mirror', False)
        self._color_aug = kwargs.get('color_augmentation', False)
        self._min_random_scale = kwargs.get('min_random_scale', 1.0)
        self._max_random_scale = kwargs.get('max_random_scale', 1.0)
        self._force_color = kwargs.get('force_color', False)
        self._phase = kwargs.get('phase', 'TRAIN')
        self._random_seed = config.GetRandomSeed()
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
        # decode
        datum = pb.Datum()
        datum.ParseFromString(serialized)
        im = np.fromstring(datum.data, np.uint8)
        if datum.encoded is True:
            im = cv2.imdecode(im, -1)
        else:
            im = im.reshape((datum.height, datum.width, datum.channels))

        # random scale
        random_scale = npr.uniform() * (
            self._max_random_scale - self._min_random_scale) \
                + self._min_random_scale
        if random_scale != 1.0:
            if sys.version_info >= (3, 0):
                im = cv2.resize(im, None, interpolation=cv2.INTER_LINEAR,
                        fx=random_scale, fy=random_scale)
            else:
                # Fuck Fuck Fuck opencv-python2, it always has a BUG
                # that leads to duplicate cuDA handles created at gpu:0
                new_shape = (
                    int(np.ceil(im.shape[1] * random_scale)),
                    int(np.ceil(im.shape[0] * random_scale)))
                im = PIL.Image.fromarray(im)
                im = im.resize(new_shape, PIL.Image.BILINEAR)
                im = np.array(im)

        # random crop
        if self._crop_size > 0:
            if self._phase == 'TRAIN':
                h_off = npr.randint(im.shape[0] - self._crop_size + 1)
                w_off = npr.randint(im.shape[1] - self._crop_size + 1)
            else:
                h_off = int((im.shape[0] - self._crop_size) / 2)
                w_off = int((im.shape[1] - self._crop_size) / 2)
            im = im[h_off : h_off + self._crop_size,
                    w_off : w_off + self._crop_size, :]

        # random mirror
        if self._mirror:
            if npr.randint(0, 2) > 0:
                im = im[:, ::-1, :]

        # gray transformation
        if self._force_color:
            if im.shape[2] == 1:
                # duplicate to 3 channels
                im = np.concatenate([im, im, im], axis=2)

        # color augmentation
        if self._color_aug:
            im = PIL.Image.fromarray(im)
            delta_brightness = npr.uniform(-0.4, 0.4) + 1.0
            delta_contrast = npr.uniform(-0.4, 0.4) + 1.0
            delta_saturation = npr.uniform(-0.4, 0.4) + 1.0
            im = PIL.ImageEnhance.Brightness(im)
            im = im.enhance(delta_brightness)
            im = PIL.ImageEnhance.Contrast(im)
            im = im.enhance(delta_contrast)
            im = PIL.ImageEnhance.Color(im)
            im = im.enhance(delta_saturation)
            im = np.array(im)

        # padding
        if self._padding > 0:
            pad_img = np.empty((im.shape[0] + 2 * self._padding,
                                im.shape[1] + 2 * self._padding, im.shape[2]),
                                dtype=im.dtype)
            pad_img.fill(self._fill_value)
            pad_img[self._padding : self._padding + im.shape[0],
                    self._padding : self._padding + im.shape[1], :] = im
            im = pad_img

        # labels
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
        npr.seed(self._random_seed)
        while True:
            serialized = self.Q_in.get()
            self.Q_out.put(self.get(serialized))