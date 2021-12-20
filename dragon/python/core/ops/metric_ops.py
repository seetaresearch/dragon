# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Metric ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema


@OpSchema.num_inputs(2)
def accuracy(inputs, axis=-1, top_k=1, ignore_index=None, **kwargs):
    """Compute the top-k accuracy.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``label``.
    axis : int, optional, default=-1
        The axis to reduce, can be negative.
    top_k : int, optional, default=1
        The top-k accuracy to compute.
    ignore_index : int, optional
        The ignored value of target.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Accuracy', inputs, axis=axis, top_k=top_k,
            ignore_index=ignore_index)
    return OpLib.add('Accuracy', inputs, axis=axis, top_k=top_k,
                     ignore_index=ignore_index, **kwargs)
