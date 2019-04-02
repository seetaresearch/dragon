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

from dragon import ops as _ops


def cumsum(x, axis=None):
    """Compute the cumulative sum along the given axis.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    axis : int, optional
        The axis to sum.

    """
    raise NotImplementedError()


def cumprod(x, axis=None):
    """Compute the cumulative product along the given axis.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    axis : int, optional
        The axis to sum.

    """
    raise NotImplementedError()


def to_one_hot(y, nb_class, **kwargs):
    """Generate a matrix where each row corresponds to the one hot encoding.

    The ``y`` should be a 1d vector.

    Parameters
    ----------
    y: Tensor
        The input tensor.
    nb_class : int
        The number of classes.

    Returns
    -------
    Tensor
        The one hot matrix.

    """
    flat_y = _ops.Flatten(y, keep_axes=1)
    return _ops.OneHot(flat_y, depth=nb_class)
