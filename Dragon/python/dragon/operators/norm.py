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

from . import *


def BatchNorm(
    inputs, axis=-1, momentum=0.9, eps=1e-5,
        use_stats=-1, mode='DEFAULT', **kwargs
):
    """Batch Normalization. `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    It follows the implementation of `Caffe`_, that scale procedure is moved to `ops.Scale(*args, **kwargs)`_.

    The number of inputs vary from ``3`` to ``4`` (``DEFAULT`` or ``CAFFE`` mode).

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, mean, var] or [input, mean, var, factor].
    axis : int
        The channel axis.
    momentum : float
        The momentum of moving average.
    eps : float
        The eps.
    use_stats : int
        Whether to use global stats. Default is ``-1`` (Auto).
    mode : str
        The moving average mode. ``DEFAULT`` or ``CAFFE``.

    Returns
    -------
    Tensor
        The output tensor, calculated as:

        |batchnorm_function|

        The ``DEFAULT`` moving average of mean/var, calculated as:

        |default_moving_average_function|

        The ``CAFFE`` moving average of mean/var, calculated as:

        |caffe_moving_average_function|

    """
    CheckInputs(inputs, 3, 4)
    arguments = ParseArguments(locals())

    if len(inputs) > 3:
        if mode != 'CAFFE':
            raise ValueError('Only the CAFFE mode will take 4 inputs.')

    output = Tensor.CreateOperator(nout=1, op_type='BatchNorm', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def BatchRenorm(
    inputs, axis=-1, momentum=0.9, eps=1e-5,
        r_max=3.0, d_max=5.0, t_delta=0.001,
            use_stats=-1, mode='DEFAULT', **kwargs
):
    """Batch Renormalization. `[Ioffe, 2017] <https://arxiv.org/abs/1702.03275>`_.

    It follows the implementation of `Caffe`_, that scale procedure is moved to `ops.Scale(*args, **kwargs)`_.

    The number of inputs vary from ``3`` to ``4`` (``DEFAULT`` or ``CAFFE`` mode).

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, mean, var, factor].
    axis : int
        The channel axis.
    momentum : float
        The momentum of moving average.
    eps : float
        The eps.
    r_max : float
        The higher bound of r.
    d_max : float
        The higher bound of d.
    t_delta : float
        The magnitude of incrementing after each iteration.
    use_stats : int
        Whether to use global stats. Default is ``-1`` (Auto).
    mode : str
        The moving average mode. ``DEFAULT`` or ``CAFFE``.

    Returns
    -------
    Tensor
        The output tensor

        |batchrenorm_function|

        The ``DEFAULT`` moving average of mean/var, calculated as:

        |default_moving_average_function|

        The ``CAFFE`` moving average of mean/var, calculated as:

        |caffe_moving_average_function|

    """
    CheckInputs(inputs, 3, 4)
    arguments = ParseArguments(locals())

    if len(inputs) > 3:
        if mode != 'CAFFE':
            raise ValueError('Only the CAFFE mode will take 4 inputs.')

    output = Tensor.CreateOperator(nout=1, op_type='BatchRenorm', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def FusedBatchNorm(
    inputs, axis=-1, momentum=0.9, eps=1e-5,
        use_stats=-1, **kwargs
):
    """Batch Normalization, with scale procedure after normalization.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, mean, var, scale, bias].
    axis : int
        The channel axis.
    momentum : float
        The momentum of moving average.
    eps : float
        The eps.
    use_stats : int
        Whether to use global stats. Default is ``-1`` (Auto).

    Returns
    -------
    Tensor
        The output tensor, calculated as:

        |batchnorm_scale_function|

        The moving average of mean/var, calculated as:

        |default_moving_average_function|

    """
    CheckInputs(inputs, 5)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='FusedBatchNorm', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def GroupNorm(inputs, group=32, axis=-1, eps=1e-5, **kwargs):
    """Group Normalization. `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    group : int
        The group size.
    axis : int
        The channel axis.
    eps : float
        The eps.

    Returns
    -------
    Tensor
        The output tensor, calculated as:

        |groupnorm_function|

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='GroupNorm', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def FusedGroupNorm(inputs, group=32, axis=-1, eps=1e-5, **kwargs):
    """Group Normalization, with scale procedure after normalization.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, scale, bias].
    group : int
        The group size.
    axis : int
        The channel axis.
    eps : float
        The eps.

    Returns
    -------
    Tensor
        The output tensor, calculated as:

        |groupnorm_scale_function|

    """
    CheckInputs(inputs, 3)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='FusedGroupNorm', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def InstanceNorm(inputs, axis=-1, eps=1e-5, **kwargs):
    """Instance Normalization. `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The channel axis.
    eps : float
        The eps.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='InstanceNorm', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def L2Norm(inputs, axis=0, num_axes=-1, eps=1e-5, mode='SUM', **kwargs):
    """L2 Normalization. `[Liu et.al, 2015] <https://arxiv.org/abs/1506.04579>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The start axis of stats region.
    num_axes : int
        The number of axes of stats region. Default is ``-1`` (Till End).
    eps : float
        The eps.
    mode : str
        The mode on computing normalizer. ``SUM`` or ``MEAN``.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='L2Norm', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output