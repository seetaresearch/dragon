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
        batch_size : int, optional, default=128
            The size of a mini-batch.
        partition : bool, optional, default=False
            Whether to partition batch for parallelism.
        prefetch : int, optional, default=5
            The prefetch count.

        """
        super(BlobFetcher, self).__init__()
        self._batch_size = kwargs.get('batch_size', 100)
        self._partition  = kwargs.get('partition', False)
        if self._partition: self._batch_size //= kwargs['group_size']
        self.Q_in = self.Q_out = None
        self.daemon = True

    def get(self):
        """Return a batch with image and label blob.

        Returns
        -------
        tuple
            The blob of image and labels.

        """
        im, labels = self.Q_in.get()
        im_blob = np.zeros(shape=([self._batch_size] + list(im.shape)), dtype=im.dtype)
        label_blob = np.zeros((self._batch_size, len(labels)), dtype=np.int64)
        for i in range(0, self._batch_size):
            im_blob[i, :, :, :], label_blob[i, :] = im, labels
            if i != self._batch_size - 1: im, labels = self.Q_in.get()
        im_blob = im_blob.astype(np.float32)
        return im_blob, label_blob

    def run(self):
        """Start the process.

        Returns
        -------
        None

        """
        while True: self.Q_out.put(self.get())