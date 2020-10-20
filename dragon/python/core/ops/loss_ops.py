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

from dragon.core.eager import context
from dragon.core.ops import activation_ops
from dragon.core.ops import loss_ops_lib
from dragon.core.ops.utils import OpSchema
from dragon.core.ops.utils import parse_args


@OpSchema.num_inputs(2)
def ctc_loss(inputs, padding_mask=-1, **kwargs):
    r"""Compute the ctc loss with batched labels.
    `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

    The shape of ``logit`` and ``label`` should be :math:`(T, N, C)`,
    :math:`(N, C)` respectively, where :math:`T` is the sequence length,
    :math:`N` is the batch size, and :math:`C` is max label length. The range
    of ``labels`` should be :math:`[1, C)`, as :math:`0` is reserved for blank.

    Use ``padding_mask`` to fill it when length is not sufficient.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``label``.
    padding_mask : int, optional, default=-1
        The mask for padding the redundant labels.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    inputs[0] = activation_ops.softmax(inputs[0], axis=2)
    op_lib = loss_ops_lib.Operator
    if context.executing_eagerly():
        raise NotImplementedError
    else:
        return op_lib.blend('CTCLoss', **args)


@OpSchema.num_inputs(1, 2)
def l1_loss(inputs, reduction='mean', **kwargs):
    r"""Compute the element-wise absolute value difference.

    The **L1Loss** function is defined as:

    .. math:: \text{L1Loss}(x, y) = |x - y|

    Examples:

    ```python
    logit = dragon.constant([1., 2., 3.])
    target = dragon.constant([0., 0., 0.])
    print(dragon.losses.l1_loss([logit, target]))  # 2.0
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``target``.
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['reduction'] = reduction.upper()
    op_lib = loss_ops_lib.L1Loss
    if context.executing_eagerly():
        return op_lib \
            .instantiate(reduction=args['reduction']) \
            .apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1, 2)
def l2_loss(inputs, reduction='mean', **kwargs):
    r"""Compute the element-wise squared error.

    The **L2Loss** function is defined as:

    .. math:: \text{L2Loss}(x, y) = (x - y)^{2}

    Examples:

    ```python
    logit = dragon.constant([1., 2., 3.])
    target = dragon.constant([0., 0., 0.])
    print(dragon.losses.l2_loss([logit, target]))  # 4.666666
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``target``.
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['reduction'] = reduction.upper()
    op_lib = loss_ops_lib.L2Loss
    if context.executing_eagerly():
        return op_lib \
            .instantiate(reduction=args['reduction']) \
            .apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2)
def nll_loss(
    inputs,
    axis=1,
    ignore_index=None,
    reduction='valid',
    **kwargs
):
    r"""Compute the negative likelihood loss with sparse labels.

    The **NLLLoss** function is defined as:

    .. math:: \text{NLLLoss}(p_{t}) = -\log(p_{t})

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``label``.
    axis : int, optional, default=1
        The reduce axis, can be negative.
    ignore_index : int, optional
        The label index to ignore.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['reduction'] = reduction.upper()
    op_lib = loss_ops_lib.NLLLoss
    if context.executing_eagerly():
        return op_lib  \
            .instantiate(
                axis=axis,
                reduction=args['reduction'],
                ignore_index=ignore_index,
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2)
def sigmoid_cross_entropy(inputs, reduction='valid', **kwargs):
    r"""Compute the sigmoid cross entropy with contiguous targets.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    logit = dragon.constant([0.1, 0.2, 0.3, 0.4])
    target = dragon.constant([0., 0., 1., 1.])
    print(dragon.losses.sigmoid_cross_entropy([logit, target]))  # 0.65247655
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``target``.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['reduction'] = reduction.upper()
    op_lib = loss_ops_lib.SigmoidCrossEntropy
    if context.executing_eagerly():
        return op_lib \
            .instantiate(reduction=args['reduction']) \
            .apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2)
def sigmoid_focal_loss(
    inputs,
    axis=1,
    alpha=0.25,
    gamma=2.,
    negative_index=None,
    reduction='valid',
    **kwargs
):
    r"""Compute the sigmoid focal loss with sparse labels.
    `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.

    The Focal loss function is defined as:

    .. math:: \text{FocalLoss}(p_{t}) = -(1 - p_{t})^{\gamma}\log(p_{t})

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``label``.
    axis : int, optional, default=1
        The reduce axis, can be negative.
    alpha : float, optional, default=0.25
        The scale factor on the rare class.
    gamma : float, optional, default=2.
        The exponential decay factor on the easy examples.
    negative_index : int, optional
        The negative class index.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['alpha'] = float(args['alpha'])
    args['gamma'] = float(args['gamma'])
    args['reduction'] = reduction.upper()
    op_lib = loss_ops_lib.SigmoidFocalLoss
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                alpha=args['alpha'],
                gamma=args['gamma'],
                negative_index=negative_index,
                reduction=args['reduction'],
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1, 2)
def smooth_l1_loss(inputs, beta=1., reduction='mean', **kwargs):
    r"""Compute the element-wise error transited from L1 and L2.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    The **SmoothL1Loss** function is defined as:

    .. math::
        \text{SmoothL1Loss}(x, y) =
            \begin{cases}
                0.5 * (x - y)^{2} / beta, & \text{ if } |x - y| < beta \\
                |x - y| - 0.5 * beta, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``target``.
    beta : float, optional, default=1.
        The transition point from L1 to L2 loss
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['beta'] = float(args['beta'])
    args['reduction'] = reduction.upper()
    op_lib = loss_ops_lib.SmoothL1Loss
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                beta=args['beta'],
                reduction=args['reduction'],
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2)
def softmax_cross_entropy(inputs, axis=1, reduction='mean', **kwargs):
    r"""Compute the softmax cross entropy with contiguous targets.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    logit = dragon.constant([[0.5, 0.5], [0.3, 0.7]])
    target = dragon.constant([[0., 1., ], [1., 0.]])
    print(dragon.losses.softmax_cross_entropy([logit, target]))  # 0.8030813
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``target``.
    axis : int, optional, default=1
        The axis to reduce, can be negative.
    reduction : {'none', 'sum', 'mean'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['reduction'] = reduction.upper()
    op_lib = loss_ops_lib.SoftmaxCrossEntropy
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                reduction=args['reduction'],
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2)
def sparse_softmax_cross_entropy(
    inputs,
    axis=1,
    ignore_index=None,
    reduction='valid',
    **kwargs
):
    r"""Compute the softmax cross entropy with sparse labels.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    logit = dragon.constant([[0.5, 0.5], [0.3, 0.7]])
    label = dragon.constant([1, 0])
    print(dragon.losses.sparse_softmax_cross_entropy([logit, label]))  # 0.8030813
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``logit`` and ``label``.
    axis : int, optional, default=1
        The axis to reduce, can be negative.
    ignore_index : int, optional
        The label index to ignore.
    reduction : {'none', 'sum', 'mean', 'valid'}, optional
        The reduction method.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['reduction'] = reduction.upper()
    op_lib = loss_ops_lib.SparseSoftmaxCrossEntropy
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                reduction=args['reduction'],
                ignore_index=ignore_index,
            ).apply(inputs)
    else:
        return op_lib.blend(**args)
