# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""The metric ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.eager import context
from dragon.core.ops import metric_ops_lib
from dragon.core.ops.utils import OpSchema
from dragon.core.ops.utils import parse_args


@OpSchema.num_inputs(2)
def accuracy(inputs, axis=-1, top_k=1, ignore_index=None, **kwargs):
    """Compute the top-k accuracy.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``label``.
    axis : int, optional, default=-1
        The axis of classes.
    top_k : int, optional, default=1
        The top-k accuracy to compute.
    ignore_index : int, optional
        The label index to ignore.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = metric_ops_lib.Accuracy
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                top_k=top_k,
                ignore_index=ignore_index,
            ).apply(inputs)
    else:
        return op_lib.blend(**args)
