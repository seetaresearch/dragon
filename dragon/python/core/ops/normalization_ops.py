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
"""Normalization ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core import distributed
from dragon.core.eager import context
from dragon.core.ops import normalization_ops_lib
from dragon.core.ops.utils import ArgHelper
from dragon.core.ops.utils import OpSchema
from dragon.core.util import nest


@OpSchema.num_inputs(5)
@ArgHelper.desc('momentum', as_target=False)
def batch_norm(
    inputs,
    axis=-1,
    momentum=0.9,
    epsilon=1e-5,
    use_stats=-1,
    **kwargs
):
    r"""Apply the batch normalization.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The running average of statistics are calculated as:

    .. math:: x_{\text{running}} = \text{momentum} * x_{\text{running}} +
                                   (1 - \text{momentum}) * x_{\text{batch}}

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``gamma``, ``beta``, ``mean`` and ``var``.
    axis : int, optional, default=-1
        The channel axis.
    momentum : Union[float, dragon.Tensor], optional
        The value to :math:`\text{momentum}`.
    epsilon : float, optional, default=1e-5
        The value to :math:`\epsilon`.
    use_stats : int, optional, default=-1
        Whether to use estimated statistics or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['epsilon'] = float(epsilon)
    op_lib = normalization_ops_lib.BatchNorm
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                epsilon=args['epsilon'],
                use_stats=use_stats,
            ).apply(inputs, args['momentum'])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(3)
def group_norm(inputs, axis=-1, group=32, epsilon=1e-5, **kwargs):
    r"""Apply the group normalization.
    `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Note that it turns out to be **InstanceNorm**, if ``group`` is  **0**,
    or **LayerNorm**, if ``group`` is **1**.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``gamma`` and ``beta``.
    axis : int, optional, default=-1
        The channel axis.
    group : int, optional, default=32
        The group size.
    epsilon : float, optional, default=1e-5
        The value to :math:`\epsilon`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['epsilon'] = float(epsilon)
    op_lib = normalization_ops_lib.GroupNorm
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                group=group,
                epsilon=args['epsilon'],
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(3)
def instance_norm(inputs, axis=-1, epsilon=1e-5, **kwargs):
    r"""Apply the instance normalization.
    `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_

    The normalization is defined as:

    .. math:: \text{out} = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``gamma`` and ``beta``.
    axis : int, optional, default=-1
        The channel axis.
    epsilon : float, optional, default=1e-5
        The value to :math:`\epsilon`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return group_norm(inputs, axis=axis, group=0, epsilon=epsilon, **kwargs)


@OpSchema.num_inputs(3)
def layer_norm(inputs, axis=-1, epsilon=1e-5, **kwargs):
    r"""Apply the layer normalization.
    `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_

    The normalization is defined as:

    .. math:: \text{out} = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``gamma`` and ``beta``.
    axis : int, optional, default=-1
        The channel axis.
    epsilon : float, optional, default=1e-5
        The value to :math:`\epsilon`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return group_norm(inputs, axis=axis, group=1, epsilon=epsilon, **kwargs)


@OpSchema.num_inputs(1)
def lp_normalize(inputs, axis=None, p=2, epsilon=1e-12, reduction='sum', **kwargs):
    r"""Apply the lp normalization.

    The **Lp-Normalization** is defined as:

    .. math:: \text{out} = \frac{x}{\max(\left\|x\right\|_{p}, \epsilon)}

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]], 'float32')

    # A negative ``axis`` is the last-k axis
    print(dragon.math.lp_normalize(x, 1))
    print(dragon.math.lp_normalize(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to compute a norm scalar
    print(dragon.math.lp_normalize(x))

    # Also, ``axis`` could be a sequence of integers
    print(dragon.math.lp_normalize(x, [0, 1]))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor :math:`x`.
    p : int, optional, default=2
        The order of the normalization.
    axis : Union[int, Sequence[int]], optional
        The axis to compute the norm.
    epsilon : float, optional, default=1e-12
        The value to :math:`\epsilon`.
    reduction : {'sum', 'mean'}, optional
        The reduction method for norm.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    if axis is None:
        args['axis'], args['num_axes'] = 0, -1
    else:
        axes = nest.flatten(axis)
        axes.sort()
        if axes[-1] != (axes[0] + len(axes) - 1):
            raise ValueError('The <axis> should be a continuous sequence.')
        args['axis'], args['num_axes'] = axes[0], len(axes)
    args['num_axes'] = kwargs.get('num_axes', args['num_axes'])
    args['epsilon'] = float(epsilon)
    args['reduction'] = reduction.upper()
    op_lib = normalization_ops_lib.LpNormalize
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                p=p,
                axis=args['axis'],
                num_axes=args['num_axes'],
                epsilon=args['epsilon'],
                reduction=args['reduction'],
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def local_response_norm(
    inputs,
    size=5,
    alpha=0.0001,
    beta=0.75,
    bias=1.,
    data_format='NCHW',
    **kwargs
):
    r"""Apply the local response normalization.
    `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

    The normalization is defined as:

    .. math::
        out_{i} = x_{i}\left(k + \frac{\alpha}{n}
                       \sum_{j=\max(0, i-n/2)}^{\min(N-1,i+n/2)}x_{j}^2
                       \right)^{-\beta}

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    size : int, optional, default=5
        The number of neighbouring channels to sum over.
    alpha : float, optional, default=0.0001
        The scale value :math:`\alpha`.
    beta : float, optional, default=0.75
        The exponent value :math:`\beta`.
    bias : float, optional, default=1.
        The bias constant :math:`k`.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    args['alpha'], args['beta'], args['bias'] = \
        float(alpha), float(beta), float(bias)
    op_lib = normalization_ops_lib.LocalResponseNorm
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                size=size,
                alpha=args['alpha'],
                beta=args['beta'],
                bias=args['bias'],
                data_format=data_format,
            ).apply([inputs])
    else:
        return op_lib.blend('LRN', **args)


@OpSchema.num_inputs(5)
@ArgHelper.desc('momentum', as_target=False)
def sync_batch_norm(
    inputs,
    axis=-1,
    momentum=0.9,
    epsilon=1e-5,
    use_stats=-1,
    process_group=None,
    **kwargs
):
    r"""Apply the batch normalization with synced statistics.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: \text{out} = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The running average of statistics are calculated as:

    .. math:: x_{\text{running}} = \text{momentum} * x_{\text{running}} +
                                   (1 - \text{momentum}) * x_{\text{batch}}

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``gamma``, ``beta``, ``mean`` and ``var``.
    axis : int, optional, default=-1
        The channel axis.
    momentum : Union[float, dragon.Tensor], optional
        The value to :math:`\text{momentum}`.
    epsilon : float, optional, default=1e-5
        The value to :math:`\epsilon`.
    use_stats : int, optional, default=-1
        Whether to use estimated statistics or not.
    process_group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['epsilon'] = float(epsilon)
    if process_group is None:
        process_group = distributed.get_group()
    if process_group is None:
        raise ValueError('<process_group> is required.')
    op_lib = normalization_ops_lib.SyncBatchNorm
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                epsilon=args['epsilon'],
                use_stats=use_stats,
                process_group=process_group,
            ).apply(inputs, args['momentum'])
    else:
        args.update(process_group.arguments)
        return op_lib.blend(**args)
