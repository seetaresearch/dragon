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
"""Loss ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema
from dragon.core.ops import activation_ops


@OpSchema.num_inputs(2)
def ctc_loss(inputs, padding_mask=-1, **kwargs):
    r"""Compute the ctc loss with batched labels.
    `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

    The shape of ``input`` and ``target`` should be :math:`(T, N, C)`,
    :math:`(N, C)` respectively, where :math:`T` is the sequence length,
    :math:`N` is the batch size, and :math:`C` is max label length. The range
    of ``labels`` should be :math:`[1, C)`, as :math:`0` is reserved for blank.

    Use ``padding_mask`` to fill it when length is not sufficient.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``input`` and ``target``.
    padding_mask : int, optional, default=-1
        The mask for padding the redundant labels.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs[0] = activation_ops.softmax(inputs[0], axis=2)
    if context.executing_eagerly():
        raise NotImplementedError
    return OpLib.add('CTCLoss', inputs, padding_mask=padding_mask, **kwargs)


@OpSchema.num_inputs(1, 2)
def l1_loss(inputs, reduction='mean', **kwargs):
    r"""Compute the loss of element-wise absolute value difference.

    The **L1Loss** function is defined as:

    .. math:: \text{L1Loss}(x, y) = |x - y|

    Examples:

    ```python
    x = dragon.constant([1., 2., 3.])
    y = dragon.constant([0., 0., 0.])
    print(dragon.losses.l1_loss([x, y]))  # 2.0
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``input`` and ``target``.
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    reduction = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute('L1Loss', inputs, reduction=reduction)
    return OpLib.add('L1Loss', inputs, reduction=reduction, **kwargs)


@OpSchema.num_inputs(min_num=1, max_num=2)
def l2_loss(inputs, reduction='mean', **kwargs):
    r"""Compute the loss of element-wise squared error.

    The **L2Loss** function is defined as:

    .. math:: \text{L2Loss}(x, y) = (x - y)^{2}

    Examples:

    ```python
    x = dragon.constant([1., 2., 3.])
    y = dragon.constant([0., 0., 0.])
    print(dragon.losses.l2_loss([x, y]))  # 4.666666
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``input`` and ``target``.
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    reduction = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute('L2Loss', inputs, reduction=reduction)
    return OpLib.add('L2Loss', inputs, reduction=reduction, **kwargs)


@OpSchema.num_inputs(2)
def nll_loss(inputs, axis=-1, ignore_index=None, reduction='valid', **kwargs):
    """Compute the loss of negative likelihood.

    Examples:

    ```python
    x = dragon.constant([[0.5, 0.5], [0.3, 0.7]])
    x = dragon.math.log(x)
    y = dragon.constant([1, 0])
    print(dragon.losses.nll_loss([x, y]))  # 0.9485599
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``input`` and ``target``.
    axis : int, optional, default=-1
        The reduction axis.
    ignore_index : int, optional
        The ignored value of target.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    reduction = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute(
            'NLLLoss', inputs, axis=axis, ignore_index=ignore_index,
            reduction=reduction)
    return OpLib.add('NLLLoss', inputs, axis=axis, ignore_index=ignore_index,
                     reduction=reduction, **kwargs)


@OpSchema.num_inputs(2)
def sigmoid_cross_entropy_loss(inputs, reduction='valid', **kwargs):
    """Compute the loss of sigmoid cross entropy.

    Examples:

    ```python
    x = dragon.constant([0.1, 0.2, 0.3, 0.4])
    y = dragon.constant([0., 0., 1., 1.])
    print(dragon.losses.sigmoid_cross_entropy_loss([x, y]))  # 0.65247655
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``input`` and ``target``.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    reduction = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute(
            'SigmoidCrossEntropyLoss', inputs, reduction=reduction)
    return OpLib.add('SigmoidCrossEntropyLoss', inputs,
                     reduction=reduction, **kwargs)


@OpSchema.num_inputs(2)
def sigmoid_focal_loss(
    inputs,
    axis=-1,
    alpha=0.25,
    gamma=2.0,
    start_index=0,
    reduction='valid',
    **kwargs
):
    """Compute the focal loss of sigmoid cross entropy.
    `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.

    Examples:

    ```python
    x = dragon.constant([[0.5, 0.5], [0.3, 0.7]])
    y = dragon.constant([1, 0])
    print(dragon.losses.sigmoid_focal_loss([x, y]))  # 0.3472295
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``input`` and ``target``.
    axis : int, optional, default=-1
        The reduction axis, can be negative.
    alpha : float, optional, default=0.25
        The scale factor to the positive classes.
    gamma : float, optional, default=2.0
        The exponential decay factor on the easy examples.
    start_index : int, optional, default=0
        The start value of target.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    alpha, gamma = float(alpha), float(gamma)
    reduction = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute(
            'SigmoidFocalLoss', inputs, axis=axis, alpha=alpha, gamma=gamma,
            start_index=start_index, reduction=reduction)
    return OpLib.add('SigmoidFocalLoss', inputs, axis=axis,
                     alpha=alpha, gamma=gamma, start_index=start_index,
                     reduction=reduction, **kwargs)


@OpSchema.num_inputs(1, 2)
def smooth_l1_loss(inputs, beta=1.0, reduction='mean', **kwargs):
    r"""Compute the loss of element-wise error transited from L1 and L2.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    The **SmoothL1Loss** function is defined as:

    .. math::
        \text{SmoothL1Loss}(x, y) =
            \begin{cases}
                0.5 * (x - y)^{2} / beta, & \text{ if } |x - y| < beta \\
                |x - y| - 0.5 * beta, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = dragon.constant([1., 2., 3.])
    y = dragon.constant([0., 0., 0.])
    print(dragon.losses.smooth_l1_loss([x, y]))  # 1.5
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``input`` and ``target``.
    beta : float, optional, default=1.0
        The transition point from L1 to L2 loss
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    beta = float(beta)
    reduction = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute(
            'SmoothL1Loss', inputs, beta=beta, reduction=reduction)
    return OpLib.add('SmoothL1Loss', inputs, beta=beta,
                     reduction=reduction, **kwargs)


@OpSchema.num_inputs(2)
def softmax_cross_entropy_loss(
    inputs,
    axis=-1,
    ignore_index=None,
    reduction='valid',
    **kwargs
):
    """Compute the loss of softmax cross entropy.

    Both sparse or dense targets are supported:

    ```python
    x = dragon.constant([[0.5, 0.5], [0.3, 0.7]])
    y1 = dragon.constant([1, 0])
    y2 = dragon.constant([[0., 1., ], [1., 0.]])
    print(dragon.losses.softmax_cross_entropy_loss([x, y1]))  # 0.8030813
    print(dragon.losses.softmax_cross_entropy_loss([x, y2]))  # Equivalent
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``input`` and ``target``.
    axis : int, optional, default=-1
        The axis to compute softmax.
    ignore_index : int, optional
        The ignored value of target.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    reduction = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute(
            'SoftmaxCrossEntropyLoss', inputs, axis=axis,
            ignore_index=ignore_index, reduction=reduction)
    return OpLib.add('SoftmaxCrossEntropyLoss', inputs, axis=axis,
                     ignore_index=ignore_index, reduction=reduction, **kwargs)
