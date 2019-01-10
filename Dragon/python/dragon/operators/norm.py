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


@OpSchema.Inputs(5)
def BatchNorm(
    inputs, axis=-1, momentum=0.9, eps=1e-5,
        use_stats=-1, **kwargs):
    """Batch Normalization. `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    We enforce the number of inputs should be *5*, i.e.,
    it is implemented into a fused version.

    However, you can still fix the *gamma* and *beta*,
    by disabling the their gradients directly.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [x, mean, var, gamma, beta].
    axis : int, optional
        The channel axis.
    momentum : float, optional, default=0.99
        The momentum of moving average.
    eps : float, optional, default=1e-5
        The eps.
    use_stats : int, optional, default=-1
        Whether to use global stats.

    Returns
    -------
    Tensor
        The output tensor, calculated as:

        |batchnorm_scale_function|

        The moving average of mean/var, calculated as:

        |default_moving_average_function|

    """
    return Tensor.CreateOperator('BatchNorm', **ParseArgs(locals()))


@OpSchema.Inputs(3)
def GroupNorm(inputs, group=32, axis=-1, eps=1e-5, **kwargs):
    """Group Normalization. `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    It turns out to be *InstanceNorm*, if ``group`` is  *0*,
    or *LayerNorm*, if ``group`` is *1*.

    We enforce the number of inputs should be *3*, i.e.,
    it is implemented into a fused version.

    However, you can still fix the *gamma* and *beta*,
    by disabling the their gradients directly.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [x, gamma, beta].
    group : int, optional, default=32
        The group size.
    axis : int, optional
        The channel axis.
    eps : float, optional, default=1e-5
        The eps.

    Returns
    -------
    Tensor
        The output tensor, calculated as:

        |groupnorm_scale_function|

    """
    return Tensor.CreateOperator('GroupNorm', **ParseArgs(locals()))


@OpSchema.Inputs(3)
def LayerNorm(inputs, axis=-1, eps=1e-5, **kwargs):
    """Layer Normalization. `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_

    We enforce the number of inputs should be *3*, i.e.,
    it is implemented into a fused version.

    However, you can still fix the *gamma* and *beta*,
    by disabling the their gradients directly.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [x, gamma, beta].
    axis : int, optional
        The channel axis.
    eps : float, optional, default=1e-5
        The eps.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('GroupNorm', group=1, **ParseArgs(locals()))


@OpSchema.Inputs(3)
def InstanceNorm(inputs, axis=-1, eps=1e-5, **kwargs):
    """Instance Normalization. `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_

    We enforce the number of inputs should be *3*, i.e.,
    it is implemented into a fused version.

    However, you can still fix the *gamma* and *beta*,
    by disabling the their gradients directly.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [x, gamma, beta].
    axis : int, optional
        The channel axis.
    eps : float, optional, default=1e-5
        The eps.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('GroupNorm', group=0, **ParseArgs(locals()))


@OpSchema.Inputs(1)
def L2Norm(inputs, axis=0, num_axes=-1, eps=1e-5, mode='SUM', **kwargs):
    """L2 Normalization. `[Liu et.al, 2015] <https://arxiv.org/abs/1506.04579>`_.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The x.
    axis : int, optional
        The start axis of stats region, can be negative.
    num_axes : int, optional, default=-1
        The number of axes of stats region.
    eps : float, optional, default=1e-5
        The eps.
    mode : {'SUM', 'MEAN'}, optional
        The mode on computing normalizer.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('L2Norm', **ParseArgs(locals()))