# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
import numpy.random as npr
from multiprocessing import Process

import dragon.config as config
from dragon.config import logger
import dragon.vm.caffe.proto.caffe_pb2 as pb

from .__init__ import GetProperty

try:
    import cv2
    import skimage.color
    import PIL.Image
except ImportError as e: pass

class DataTransformer(Process):
    def __init__(self, **kwargs):
        super(DataTransformer, self).__init__()

        # mean subtraction
        self._mean_value = GetProperty(kwargs, 'mean_value', [])
        self._mean_file = GetProperty(kwargs, 'mean_file', '')
        if self._mean_file:
            self._mean_value = cv2.imread(self._mean_file).astype(np.float32)

        # value range scale
        self._scale = GetProperty(kwargs, 'scale', 1.0)

        # augmentation
        self._crop_size = GetProperty(kwargs, 'crop_size', 0)
        self._mirror = GetProperty(kwargs, 'mirror', False)
        self._color_aug = GetProperty(kwargs, 'color_augmentation', False)
        self._min_random_scale = GetProperty(kwargs, 'min_random_scale', 1.0)
        self._max_random_scale = GetProperty(kwargs, 'max_random_scale', 1.0)

        # utility
        self._force_gray = GetProperty(kwargs, 'force_gray', False)
        self._phase = GetProperty(kwargs, 'phase', 'TRAIN')
        self._random_seed = config.GetRandomSeed()
        self.Q_in = self.Q_out = None
        self.daemon = True

        def cleanup():
            logger.info('Terminating DataTransformer......')
            self.terminate()
            self.join()
        import atexit
        atexit.register(cleanup)

    def transform_image_label(self, serialized):
       datum = pb.Datum()
       datum.ParseFromString(serialized)
       im = np.fromstring(datum.data, np.uint8)
       if datum.encoded is True:
           im = cv2.imdecode(im, -1)
       else:
           im = im.reshape((datum.height, datum.width, datum.channels))

       # handle scale
       random_scale = npr.uniform() * (self._max_random_scale - self._min_random_scale) \
                            + self._min_random_scale
       if random_scale != 1.0:
           new_shape = (int(im.shape[1] * random_scale), int(im.shape[0] * random_scale))
           im = PIL.Image.fromarray(im)
           im = im.resize(new_shape, PIL.Image.BILINEAR)
           im = np.array(im)

       # handle gray
       if not self._force_gray:
            if im.shape[2] == 1:
                im = np.concatenate([im, im, im], axis=2) # copy to 3 channels

       # handle crop
       h_off = w_off = 0
       if self._crop_size > 0:
            if self._phase == 0:
                h_off = npr.randint(im.shape[0] - self._crop_size + 1)
                w_off = npr.randint(im.shape[1] - self._crop_size + 1)
            else:
                h_off = (im.shape[0] - self._crop_size) / 2
                w_off = (im.shape[1] - self._crop_size) / 2
            im = im[h_off : h_off + self._crop_size, w_off : w_off + self._crop_size, :]

       # handle mirror
       if self._mirror:
            if npr.randint(0, 2) > 0:
                im = im[:, ::-1, :]

       # handle color augmentation
       if self._color_aug:
            if npr.randint(0, 2) > 0:
                im = im[:, :, ::-1]  # BGR -> RGB
                im = skimage.color.rgb2hsv(im)
                h, s, v = np.split(im, 3, 2)
                delta_h = npr.uniform() * 0.2 - 0.1
                delta_s = npr.uniform() * 0.2 - 0.1
                delta_v = npr.uniform() * 0.2 - 0.1
                h = np.clip(h + delta_h, 0, 1)
                s = np.clip(s + delta_s, 0, 1)
                v = np.clip(v + delta_v, 0, 1)
                im = np.concatenate([h, s, v], axis=2)
                im = skimage.color.hsv2rgb(im)
                im = im[:, :, ::-1] * np.array([255])

       im = im.astype(np.float32, copy=False)

       # handle mean subtraction
       if len(self._mean_value) > 0:
           if self._mean_file:
               im = im - self._mean_value[h_off : h_off + self._crop_size, w_off : w_off + self._crop_size, :]
           else: im = im - self._mean_value

       # handle range scale
       if self._scale != 1.0:
            im = im * self._scale

       return im, [datum.label]

    def run(self):
        npr.seed(self._random_seed)
        while True:
            serialized = self.Q_in.get()
            self.Q_out.put(self.transform_image_label(serialized))


