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

from dragon.utils import vision as _vision
from dragon.core import workspace as _workspace


class MiniBatchOp(object):
    """Form a mini-batch based on `dragon.utils.vision`_ package."""

    def setup(self, inputs, outputs):
        """Setup for params or options.

        Parameters
        ----------
        inputs : sequence of str
            The name of inputs.
        outputs : sequence of str
            The name of outputs.

        Returns
        -------
        None

        """
        kwargs = eval(self.param_str)
        self._data_batch = _vision.DataBatch(**kwargs)

    def run(self, inputs, outputs):
        """Run method, i.e., forward pass.

        Parameters
        ----------
        inputs : sequence of str
            The name of inputs.
        outputs : sequence of str
            The name of outputs.

        Returns
        -------
        None

        """
        blobs = self._data_batch.get()
        for idx, blob in enumerate(blobs):
            _workspace.FeedTensor(outputs[idx], blob)