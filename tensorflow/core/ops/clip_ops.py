# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/clip_ops.py>
#
# ------------------------------------------------------------
"""Clip ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import math_ops


def clip_by_value(t, clip_value_min, clip_value_max, name=None):
    r"""Compute the clipped input according to the given bounds.

    .. math:: \text{out} = \min(\max(x, \text{low}), \text{high})

    Examples:

    ```python
    x = tf.constant([-2, -1, 0, 1, 2])
    print(tf.clip_by_value(x, clip_value_min=0, clip_value_max=1))
    ```

    Parameters
    ----------
    t : dragon.Tensor
        The tensor :math:`x`.
    clip_value_min : number, optional
        The value to :math:`\text{low}`.
    clip_value_max : number, optional
        The value to :math:`\text{high}`.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.clip(t, clip_value_min, clip_value_max, name=name)
