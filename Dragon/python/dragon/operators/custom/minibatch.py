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

import dragon.core.workspace as ws
from dragon.io.data_batch import DataBatch


class MiniBatchOp(object):
    """
    How to form a minibatch based on `dragon.io`_ package.
    """
    def setup(self, inputs, outputs):
        """Setup for params or options.

        Parameters
        ----------
        inputs : list of str
            Indicating the name of input tensors.
        outputs : list of str
            Indicating the name of output tensors.

        Returns
        -------
        None

        """
        kwargs = eval(self.param_str)
        self._data_batch = DataBatch(**kwargs)


    def run(self, inputs, outputs):
        """Run method, i.e., forward pass.

        Parameters
        ----------
        inputs : list of str
            Indicating the name of input tensors.
        outputs : list of str
            Indicating the name of output tensors.

        Returns
        -------
        None

        """
        blobs = self._data_batch.get()
        for idx, blob in enumerate(blobs):
            ws.FeedTensor(outputs[idx], blob)