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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import init_ops


def random_normal(
    shape,
    mean=0,
    stddev=1,
    dtype='float32',
    seed=None,
    name=None,
):
    r"""Return a tensor initialized from normal distribution.

    The **Normal** distribution is defined as:

    .. math:: X \sim N(\mu, \sigma)

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The shape of the tensor.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    stddev : number, optional, default=1
        The value to :math:`\sigma`.
    dtype : str, optional
        The optional data type.
    seed : int, optional
        The optional random seed.
    name : str, optional
        A optional name for the operation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    _, dtype, init_fn = seed, str(dtype), init_ops.random_normal
    return init_fn(shape, mean, stddev, dtype=dtype, name=name)


def random_uniform(
    shape,
    minval=0,
    maxval=1,
    dtype='float32',
    seed=None,
    name=None,
):
    r"""Return a tensor initialized from the uniform distribution.

    The **Uniform** distribution is defined as:

    .. math:: X \sim U(\alpha, \beta)

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The shape of the tensor.
    minval : number, optional, default=0
        The value to :math:`\alpha`.
    maxval : number, optional, default=1
        The value to :math:`\beta`.
    dtype : str, optional
        The optional data type.
    seed : int, optional
        The optional random seed.
    name : str, optional
        A optional name for the operation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    _, dtype, init_fn = seed, str(dtype), init_ops.random_uniform
    return init_fn(shape, minval, maxval, dtype=dtype, name=name)


def truncated_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype='float32',
    seed=None,
    name=None,
):
    r"""Return a tensor initialized from the truncated normal distribution.

    The **TruncatedNormal** distribution is defined as:

    .. math::
        X \sim TN(\mu, \sigma, \mu - 2\sigma, \mu + 2\sigma)

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The shape of the tensor.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    stddev : number, optional, default=1
        The value to :math:`\sigma`.
    dtype : str, optional
        The optional data type.
    seed : int, optional
        The optional random seed.
    name : str, optional
        A optional name for the operation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    _, dtype, init_fn = seed, str(dtype), init_ops.truncated_normal
    return init_fn(shape, mean, stddev, dtype=dtype, name=name)
