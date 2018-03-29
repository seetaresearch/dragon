# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

import numpy as np
from multiprocessing import Process

from .utils import GetProperty

class BlobFetcher(Process):
    """
    BlobFetcher is deployed to queue blobs from `DataTransformer`_.

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

        """
        super(BlobFetcher, self).__init__()
        self._batch_size = GetProperty(kwargs, 'batch_size', 100)
        self._partition  = GetProperty(kwargs, 'partition', False)
        if self._partition:
            self._batch_size = int(self._batch_size / kwargs['group_size'])
        self.Q_in = self.Q_out = None
        self.daemon = True

    def im_list_to_blob(self):
        """Get image and label blobs.

        Returns
        -------
        tuple
            The blob of image and labels.

        """
        datum = self.Q_in.get()
        im_blob = []
        label_blob = np.zeros((self._batch_size, len(datum[1])), dtype=np.float32) \
                        if len(datum) > 1 else None
        for i in range(0, self._batch_size):
            im_blob.append(datum[0])
            if label_blob is not None: label_blob[i, :] = datum[1]
            if i != self._batch_size - 1: datum = self.Q_in.get()
        channel_swap = (0, 3, 1, 2)
        im_blob = np.array(im_blob, dtype=np.float32)
        im_blob = im_blob.transpose(channel_swap)
        return (im_blob, label_blob)

    def run(self):
        """Start the process.

        Returns
        -------
        None

        """
        while True:
            self.Q_out.put(self.im_list_to_blob())