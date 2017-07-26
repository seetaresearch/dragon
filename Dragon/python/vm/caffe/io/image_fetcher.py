# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import os
import numpy as np
import numpy.random as npr
from multiprocessing import Process

import dragon.config as config
import dragon.core.mpi as mpi

from __init__ import GetProperty

class Datum(object):
    def __init__(self):
        self._file = ''
        self._label = None

class ImageReader(Process):
    def __init__(self, **kwargs):
        super(ImageReader, self).__init__()
        self._shuffle = GetProperty(kwargs, 'shuffle', False)
        self._force_gray = GetProperty(kwargs, 'force_gray', False)
        self._source = GetProperty(kwargs, 'source', '')
        self._mean_value = GetProperty(kwargs, 'mean_value', [])
        self._scale = GetProperty(kwargs, 'scale', 1.0)
        self._mirror = GetProperty(kwargs, 'mirror', False)
        self._phase = GetProperty(kwargs, 'phase', 'TRAIN')
        self._crop_size = GetProperty(kwargs, 'crop_size', 0)
        self._step_val = GetProperty(kwargs, 'step', 1)
        self._random_seed = config.GetRandomSeed()
        self._indices = []
        self._cur_idx = 0
        if mpi.is_init():
            idx, group = mpi.allow_parallel()
            if idx != -1: # valid data parallel
                rank = mpi.rank()
                self._random_seed += rank # for shuffle
                for i, node in enumerate(group):
                    if rank == node: self._cur_idx = i
                if not kwargs.has_key('step'):
                    self._step_val = len(group)
        self._Q = None
        self.ParseImageSet()
        self.daemon = True
        def cleanup():
            print 'Terminating DataReader......'
            self.terminate()
            self.join()
        import atexit
        atexit.register(cleanup)

    def ParseImageSet(self):
        if not os.path.exists(self._source):
            raise RuntimeError('DataReader found the source does not exist')
        with open(self._source) as f:
            for line in f:
                content = line.split()
                item = Datum()
                item._file = content[0]
                if len(content) > 1:
                    item._label = tuple(content[idx] for idx in xrange(1, len(content)))
                else: item._label = None
                self._indices.append(item)

    def load_image(self, index):
        import PIL.Image as Image
        filepath = os.path.join(self._source, self._indices[index]._file)
        assert os.path.exists(filepath)
        im = Image.open(filepath)
        im = np.array(im, dtype=np.float32)
        if len(im.shape) < 3: im = im[:, :, np.newaxis]
        if self._force_gray: im = im[:, :, -1, np.newaxis]
        else:
            if im.shape[2] == 1:
                # copy to 3 channels
                im = np.concatenate([im, im, im], axis=2)
            else: im = im[:, :, ::-1] # RGB -> BGR

        # handle crop
        if self._crop_size > 0:
            assert im.shape[0] >= self._crop_size
            assert im.shape[1] >= self._crop_size
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

        # handle mean value
        if len(self._mean_value) > 0:
            im = im - self._mean_value

        # handle scale
        if self._scale != 1.0:
            im = im * self._scale

        return im

    def load_image_label(self, index):
        im = self.load_image(index)
        label = self._indices[index]._label
        if label is not None: return (im, label)
        else: return [im]

    def run(self):
        npr.seed(self._random_seed)
        while True:
            self._Q.put(self.load_image_label(self._cur_idx))
            if self._shuffle:
                self._cur_idx = npr.randint(0, len(self._indices))
            else: self._cur_idx = (self._cur_idx + self._step_val) % len(self._indices)

