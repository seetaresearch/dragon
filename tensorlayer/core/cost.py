# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import loss_ops


def cross_entropy(output, target, name=None):
    """Compute the cost of softmax cross entropy.

    Parameters
    ----------
    output : dragon.Tensor
        The output tensor.
    target : dragon.Tensor
        The target tensor.
    name : str, optional
        The  operator name.

    Returns
    -------
    dragon.Tensor
        The cost tensor.

    """
    return loss_ops.softmax_cross_entropy_loss(
        [output, target], reduction='mean', name=name)
