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
from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema


@OpSchema.num_inputs(5)
@OpSchema.convert_arg('momentum', as_target=False)
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

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    The running average of statistics are calculated as:

    .. math:: x_{\text{running}} = \text{momentum} * x_{\text{running}}
                                   + (1 - \text{momentum}) * x_{\text{batch}}

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
    args = OpSchema.parse_args(locals())
    args['epsilon'] = float(epsilon)
    if context.executing_eagerly():
        return OpLib.execute(
            'BatchNorm', inputs, axis=axis, epsilon=args['epsilon'],
            use_stats=use_stats, momentum=args['momentum'])
    return OpLib.add('BatchNorm', **args)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg('perm')
def channel_norm(
    inputs,
    mean,
    std,
    axis=-1,
    dtype='float32',
    perm=None,
    **kwargs
):
    """Apply the normalization to each channel of input.

    :attr:`axis` can be negative:

    ```python
    m = s = (1., 1., 1.)
    x = dragon.constant([1, 2, 3])
    print(dragon.nn.channel_norm(x, m, s, axis=0))   # [0., 1., 2.]
    print(dragon.nn.channel_norm(x, m, s, axis=-1))  # Equivalent
    ```

    If :attr:`perm` provided, :attr:`axis` is selected from the output layout:

    ```python
    m, s = (1., 2., 3.), (1., 1., 1.)
    x = dragon.constant([[1, 2, 3]])
    # Provided 3 values to normalize the last axis
    # with length 1, only the first value will be taken
    print(dragon.nn.channel_norm(x, m, s, perm=(1, 0)))  # [[0.], [1.], [2.]]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    mean : Sequence[float], required
        The mean to subtract.
    std : Sequence[float], required
        The standard deviation to divide.
    axis : int, optional, default=-1
        The channel axis.
    dtype : str, optional, default='float32'
        The output data type.
    perm : Sequence[Union[int, dragon.Tensor]], optional
        The output permutation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'ChannelNorm', inputs,
            axis=axis, mean=mean, std=std, dtype=dtype,
            ndim=len(args['perm']) if perm is not None else 0,
            perm=args['perm'])
    return OpLib.add('ChannelNorm', **args)


@OpSchema.num_inputs(3)
def group_norm(inputs, axis=-1, group=0, epsilon=1e-5, **kwargs):
    r"""Apply the group normalization.
    `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    :attr:`group` could be zero to apply the instance normalization:

    ```python
    gamma, beta = dragon.ones((3,)), dragon.zeros((3,))
    x = dragon.constant([[1., 2., 3.], [4., 5., 6.]], dtype=gamma.dtype)
    y = dragon.nn.group_norm([x, gamma, beta], group=0)
    print(y)  # [[0., 0., 0.], [0., 0., 0.]]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``gamma`` and ``beta``.
    axis : int, optional, default=-1
        The channel axis.
    group : int, optional, default=0
        The group size.
    epsilon : float, optional, default=1e-5
        The value to :math:`\epsilon`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    epsilon = float(epsilon)
    if context.executing_eagerly():
        return OpLib.execute(
            'GroupNorm', inputs, axis=axis, group=group, epsilon=epsilon)
    return OpLib.add('GroupNorm', inputs, axis=axis,
                     group=group, epsilon=epsilon, **kwargs)


@OpSchema.num_inputs(3)
def instance_norm(inputs, axis=-1, epsilon=1e-5, **kwargs):
    r"""Apply the instance normalization.
    `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

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

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``gamma`` and ``beta``.
    axis : int, optional, default=-1
        The start axis of normalized dimensions.
    epsilon : float, optional, default=1e-5
        The value to :math:`\epsilon`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    epsilon = float(epsilon)
    if context.executing_eagerly():
        return OpLib.execute(
            'LayerNorm', inputs, axis=axis, epsilon=epsilon)
    return OpLib.add('LayerNorm', inputs, axis=axis, epsilon=epsilon, **kwargs)


@OpSchema.num_inputs(1)
def lp_norm(
    inputs,
    axis=-1,
    end_axis=None,
    p=2,
    epsilon=1e-12,
    reduction='sum',
    **kwargs
):
    r"""Apply the lp normalization.

    The normalization is defined as:

    .. math:: y = \frac{x}{\max(\left\|x\right\|_{p}, \epsilon)}

    :attr:`axis` could be negative:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]], 'float32')
    # A negative axis is the last-k axis
    print(dragon.nn.lp_norm(x, 1))
    print(dragon.nn.lp_norm(x, -1))  # Equivalent
    ```

    More than one axis could be specified to reduce:

    ```python
    # Along the continuous axes: [axis, end_axis]
    print(dragon.nn.lp_norm(x, axis=0, end_axis=1))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor :math:`x`.
    p : int, optional, default=2
        The order of the normalization.
    axis : int, optional, default=-1
        The first axis to reduce.
    end_axis : int, optional
        The last axis to reduce.
    epsilon : float, optional, default=1e-12
        The value to :math:`\epsilon`.
    reduction : {'sum', 'mean'}, optional
        The reduction method for norm.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    epsilon = float(epsilon)
    reduction = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute(
            'LpNorm', inputs, p=p, axis=axis, end_axis=end_axis,
            epsilon=epsilon, reduction=reduction)
    return OpLib.add('LpNorm', inputs, p=p, axis=axis, end_axis=end_axis,
                     epsilon=epsilon, reduction=reduction, **kwargs)


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
        y_{i} = x_{i}\left(k + \frac{\alpha}{n}
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
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    alpha, beta, bias = float(alpha), float(beta), float(bias)
    if context.executing_eagerly():
        return OpLib.execute(
            'LRN', inputs, size=size, alpha=alpha, beta=beta,
            bias=bias, data_format=data_format)
    return OpLib.add('LRN', inputs, size=size, alpha=alpha, beta=beta,
                     bias=bias, data_format=data_format, **kwargs)


@OpSchema.num_inputs(5)
@OpSchema.convert_arg('momentum', as_target=False)
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

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    The running average of statistics are calculated as:

    .. math:: x_{\text{running}} = \text{momentum} * x_{\text{running}}
                                   + (1 - \text{momentum}) * x_{\text{batch}}

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
    args = OpSchema.parse_args(locals())
    args['epsilon'] = float(epsilon)
    if process_group is None:
        process_group = distributed.get_group()
    if process_group is None:
        raise ValueError('<process_group> is required.')
    if context.executing_eagerly():
        return OpLib.execute(
            'SyncBatchNorm', inputs, axis=axis, epsilon=args['epsilon'],
            use_stats=use_stats, momentum=args['momentum'],
            **process_group.arguments)
    args.pop('process_group')
    args.update(process_group.arguments)
    return OpLib.add('SyncBatchNorm', **args)
