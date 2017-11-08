# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
import numpy.random as npr
from multiprocessing import Process

import dragon.config as config
import dragon.vm.caffe.proto.caffe_pb2 as pb

from .utils import GetProperty

try:
    import cv2
    import PIL.Image
    import PIL.ImageEnhance
except ImportError as e: pass

class DataTransformer(Process):
    """
    DataTransformer is deployed to queue transformed images from `DataReader`_.

    Nearly all common image augmentation methods are supported.
    """
    def __init__(self, **kwargs):
        """Construct a ``DataTransformer``.

        Parameters
        ----------
        mean_values : list
            The mean value of each image channel.
        scale : float
            The scale performed after mean subtraction. Default is ``1.0``.
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
        self._mean_values = GetProperty(kwargs, 'mean_values', [])
        self._scale = GetProperty(kwargs, 'scale', 1.0)
        self._padding = GetProperty(kwargs, 'padding', 0)
        self._fill_value = GetProperty(kwargs, 'fill_value', 127)
        self._crop_size = GetProperty(kwargs, 'crop_size', 0)
        self._mirror = GetProperty(kwargs, 'mirror', False)
        self._color_aug = GetProperty(kwargs, 'color_augmentation', False)
        self._min_random_scale = GetProperty(kwargs, 'min_random_scale', 1.0)
        self._max_random_scale = GetProperty(kwargs, 'max_random_scale', 1.0)
        self._force_color = GetProperty(kwargs, 'force_color', False)
        self._phase = GetProperty(kwargs, 'phase', 'TRAIN')
        self._random_seed = config.GetRandomSeed()
        self.Q_in = self.Q_out = None
        self.daemon = True

        def cleanup():
            from dragon.config import logger
            logger.info('Terminating DataTransformer......')
            self.terminate()
            self.join()
        import atexit
        atexit.register(cleanup)

    def transform_image_labels(self, serialized):
        """Get image and labels from a serialized str.

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
        random_scale = npr.uniform() * (self._max_random_scale - self._min_random_scale) \
                            + self._min_random_scale
        if random_scale != 1.0:
           new_shape = (int(im.shape[1] * random_scale), int(im.shape[0] * random_scale))
           im = PIL.Image.fromarray(im)
           im = im.resize(new_shape, PIL.Image.BILINEAR)
           im = np.array(im)

        # random crop
        h_off = w_off = 0
        if self._crop_size > 0:
            if self._phase == 'TRAIN':
                h_off = npr.randint(im.shape[0] - self._crop_size + 1)
                w_off = npr.randint(im.shape[1] - self._crop_size + 1)
            else:
                h_off = (im.shape[0] - self._crop_size) / 2
                w_off = (im.shape[1] - self._crop_size) / 2
            im = im[h_off : h_off + self._crop_size, w_off : w_off + self._crop_size, :]

        # random mirror
        if self._mirror:
            if npr.randint(0, 2) > 0:
                im = im[:, ::-1, :]

        # gray transformation
        if self._force_color:
            if im.shape[2] == 1:
                im = np.concatenate([im, im, im], axis=2) # duplicate to 3 channels

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

        im = im.astype(np.float32, copy=False)

        # mean subtraction
        if len(self._mean_values) > 0:
            im = im - self._mean_values

        # numerical scale
        if self._scale != 1.0:
             im = im * self._scale

        return im, [datum.label]

    def run(self):
        """Start the process.

        Returns
        -------
        None

        """
        npr.seed(self._random_seed)
        while True:
            serialized = self.Q_in.get()
            self.Q_out.put(self.transform_image_labels(serialized))