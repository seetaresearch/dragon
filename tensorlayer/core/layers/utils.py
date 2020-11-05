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

from dragon.core.ops import array_ops


def get_act_str(fn):
    """
    Get the string representation of a function.

    Args:
        fn: (str): write your description
    """
    return fn.__name__ if fn is not None else 'No Activation'


def flatten_reshape(variable, name='flatten'):
    """Reshape a high-dimension vector input.

    Parameters
    ----------
    variable : dragon.Tensor
        The variable or tensor to be flatten.
    name : str, optional
        The optional operator name.

    Returns
    -------
    dragon.Tensor
        The output tensor

    """
    return array_ops.flatten(variable, axis=1, name=name)


def list_remove_repeat(x):
    """Remove the repeated items in a sequence."""
    y = []
    for i in x:
        if i not in y:
            y.append(i)
    return y
