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

import numpy as np
import numpy.random as npr
from multiprocessing import Process

import dragon.config as config
import dragon.vm.caffe.proto.caffe_pb2 as pb

try:
    import cv2
except ImportError as e:
    print('Failed to import cv2. Error: {0}'.format(str(e)))


class DataTransformer(Process):
    """DataTransformer is deployed to queue transformed images from `DataReader`_.

    Nearly all common image augmentation methods are supported.

    """
    def __init__(self, transform=None, color_space='RGB', pack=False, **kwargs):
        """Construct a ``DataTransformer``.

        Parameters
        ----------
        transform : lambda
            The transforms.
        color_space : str
            The color space.
        pack : boolean
            Pack the images automatically.

        """
        super(DataTransformer, self).__init__()
        self.transform = transform
        self.color_space = color_space
        self.pack = pack
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
        if datum.channels == 3 and \
            self.color_space == 'RGB': im = im[:, :, ::-1]

        # labels
        labels = []
        if len(datum.labels) > 0: labels.extend(datum.labels)
        else: labels.append(datum.label)
        return self.transform(im), labels

    def run(self):
        """Start the process.

        Returns
        -------
        None

        """
        npr.seed(self._random_seed)
        while True:
            serialized = self.Q_in.get()
            im, label = self.get(serialized)
            if len(im.shape) == 4 and not self.pack:
                for ix in range(im.shape[0]):
                    self.Q_out.put((im[ix], label))
            else:
                if len(im.shape) == 3 and self.pack:
                    im = np.expand_dims(im, axis=0)
                self.Q_out.put((im, label))