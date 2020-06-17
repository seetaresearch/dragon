from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import loss_ops


def sparse_softmax_crossentropy(
    logit,
    target,
    axis=1,
    reduction='valid',
    ignore_index=-1,
    name=None,
):
    r"""Compute the softmax cross entropy with sparse labels.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Parameters
    ----------
    logit : dragon.Tensor
        The tensor ``logit``.
    target : dragon.Tensor
        The tensor ``target``.
    axis : int, optional, default=1
        The axis to apply softmax, can be negative.
    reduction : {'none', 'sum', 'mean', 'batch_size', 'valid'}, optional
        The reduction method.
    ignore_index : int, optional, default=-1
        The label index to ignore.
    name : str, optional
        The optional operator name.

    Returns
    -------
    dragon.Tensor
        The loss.

    """
    return loss_ops.sparse_softmax_cross_entropy(
        [logit, target],
        axis=axis,
        reduction=reduction,
        ignore_index=ignore_index,
        name=name,
    )
