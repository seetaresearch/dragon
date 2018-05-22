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
from multiprocessing import Process


class BlobFetcher(Process):
    """BlobFetcher is deployed to queue blobs from `DataTransformer`_.

    It is supported to form ``NCHW`` image blobs and ``1D`` label blobs.

    """
    def __init__(self, **kwargs):
        """Construct a ``BlobFetcher``.

        Parameters
        ----------
        batch_size : int
            The size of a training batch.
        partition : boolean
            Whether to partition batch. Default is ``False``.
        prefetch : int
            The prefetch count. Default is ``5``.
        mean_values : list
            The mean value of each image channel.
        scale : float
            The scale performed after mean subtraction. Default is ``1.0``.

        """
        super(BlobFetcher, self).__init__()
        self._batch_size = kwargs.get('batch_size', 100)
        self._partition  = kwargs.get('partition', False)
        self._mean_values = kwargs.get('mean_values', [])
        self._scale = kwargs.get('scale', 1.0)
        if self._partition:
            self._batch_size = int(self._batch_size / kwargs['group_size'])
        self.Q_in = self.Q_out = None
        self.daemon = True

    def get(self):
        """Return a batch with image and label blob.

        Returns
        -------
        tuple
            The blob of image and labels.

        """
        # fill blobs
        im, labels = self.Q_in.get()
        im_blob = np.zeros(shape=([self._batch_size] + list(im.shape)), dtype=np.uint8)
        label_blob = np.zeros((self._batch_size, len(labels)),  dtype=np.int64)
        for ix in range(0, self._batch_size):
            im_blob[ix, :, :, :], label_blob[ix, :] = im, labels
            if ix != self._batch_size - 1: im, labels = self.Q_in.get()

        # mean subtraction & numerical scale
        im_blob = im_blob.astype(np.float32)
        if len(self._mean_values) > 0:
            im_blob -= self._mean_values
        if self._scale != 1.0:
            im_blob *= self._scale

        return im_blob.transpose((0, 3, 1, 2)), label_blob

    def run(self):
        """Start the process.

        Returns
        -------
        None

        """
        while True:
            self.Q_out.put(self.get())