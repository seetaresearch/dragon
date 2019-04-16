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
from .activation import Softmax


@OpSchema.Inputs(2)
def NLLLoss(
    inputs,
    axis=1,
    normalization='VALID',
    ignore_labels=(),
    **kwargs
):
    """Compute the negative likelihood loss with sparse labels.

    **Type Constraints**:

    * logits (*float16*, *float32*)

    * labels (*float32*, *int64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [logits, labels].
    axis : int, optional
        The axis to apply softmax, can be negative.
    normalization : {'UNIT', 'FULL', 'VALID', 'BATCH_SIZE', 'NONE'}, optional
        The method of normalization.
    ignore_labels : sequence of int, optional, default=()
        The label id to ignore.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set ``normalization`` to *UNIT* will return the unreduced losses.

    """
    return Tensor.CreateOperator('NLLLoss', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def SparseSoftmaxCrossEntropy(
    inputs,
    axis=1,
    normalization='VALID',
    ignore_labels=(),
    **kwargs
):
    """Compute the softmax cross entropy with sparse labels.

    **Type Constraints**:

    * logits (*float16*, *float32*)

    * labels (*float32*, *int64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [logits, labels].
    axis : int, optional
        The axis to apply softmax, can be negative.
    normalization : {'UNIT', 'FULL', 'VALID', 'BATCH_SIZE', 'NONE'}, optional
        The method of normalization.
    ignore_labels : sequence of int, optional, default=()
        The label id to ignore.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set ``normalization`` to *UNIT* will return the unreduced losses.

    """
    return Tensor.CreateOperator('SparseSoftmaxCrossEntropy', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def SigmoidCrossEntropy(inputs, normalization='VALID', **kwargs):
    """Compute sigmoid cross entropy with given logits and targets.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [logits, targets].
    normalization : {'UNIT', 'FULL', 'VALID', 'BATCH_SIZE', 'NONE'}, optional
        The method of normalization.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set ``normalization`` to *UNIT* will return the unreduced losses.

    """
    return Tensor.CreateOperator('SigmoidCrossEntropy', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def SoftmaxCrossEntropy(inputs, axis=1, normalization='FULL', **kwargs):
    """Compute the softmax cross entropy with given logits and one-hot labels.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [logits, labels].
    axis : int, optional
        The axis to apply softmax, can be negative.
    normalization : {'UNIT', 'FULL', 'BATCH_SIZE', 'NONE'}, optional
        The method of normalization.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set ``normalization`` to *UNIT* will return the unreduced losses.

    """
    return Tensor.CreateOperator('SoftmaxCrossEntropy', **ParseArgs(locals()))


@OpSchema.Inputs(2, 4)
def SmoothL1Loss(inputs, beta=1.0, normalization='BATCH_SIZE', **kwargs):
    """Compute the smoothed L1 loss. `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    Note that the ``beta`` is represented as  |smooth_l1_beta| following the original paper.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [input, targets] + [inside_w] + [outside_w].
    beta : float, optional
        The transition point from L1 to L2 loss
    normalization : {'FULL', 'BATCH_SIZE', 'NONE'}, optional
        The method of normalization.

    Returns
    -------
    Tensor
        The loss.

    """
    return Tensor.CreateOperator('SmoothL1Loss', **ParseArgs(locals()))


@OpSchema.Inputs(1, 3)
def L1Loss(inputs, scale=1., normalization='BATCH_SIZE', **kwargs):
    """Compute the L1 loss.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [x] + [targets] + [inside_w].
    scale : float, optional
        The scale factor applying on the reduced loss.
    normalization : {'FULL', 'BATCH_SIZE', 'NONE'}, optional
        The method of normalization.

    Returns
    -------
    Tensor
        The loss, calculated as:  |l1_loss_function|

    """
    return Tensor.CreateOperator('L1Loss', **ParseArgs(locals()))


@OpSchema.Inputs(1, 3)
def L2Loss(inputs, scale=1., normalization='BATCH_SIZE', **kwargs):
    """Compute the L2 loss.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [x] + [targets] + [inside_w].
    scale : float, optional
        The scale factor applying on the reduced loss.
    normalization : {'FULL', 'BATCH_SIZE', 'NONE'}, optional
        The method of normalization.

    Returns
    -------
    Tensor
        The loss, calculated as:  |l2_loss_function|

    """
    return Tensor.CreateOperator('L2Loss', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def SigmoidFocalLoss(
    inputs,
    axis=1,
    normalization='VALID',
    alpha=0.25,
    gamma=2.0,
    neg_id=0,
    **kwargs
):
    """Compute the sigmoid focal loss with sparse labels. `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [input, labels].
    axis : int, optional
        The axis to apply softmax, can be negative.
    normalization : {'UNIT', 'FULL', 'VALID', 'BATCH_SIZE', 'NONE'}, optional
        The method of normalization.
    alpha : float, optional, default=0.25
        The scale factor on the rare class.
    gamma : float, optional, default=2.0
        The exponential decay factor on the easy examples.
    neg_id : int, optional, default=0
        The negative id.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set ``normalization`` to *UNIT* will return the unreduced losses.

    """
    return Tensor.CreateOperator('SigmoidFocalLoss', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def SoftmaxFocalLoss(
    inputs,
    axis=1,
    normalization='VALID',
    ignore_labels=(),
    alpha=0.25,
    gamma=2.0,
    neg_id=0,
    **kwargs
):
    """Compute the softmax focal loss with sparse labels. `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [input, labels].
    axis : int, optional
        The axis to apply softmax, can be negative.
    normalization : {'UNIT', 'FULL', 'VALID', 'BATCH_SIZE', 'NONE'}, optional
        The method of normalization.
    ignore_labels : sequence of int, optional, default=()
        The label id to ignore.
    alpha : float, optional, default=0.25
        The scale factor on the rare class.
    gamma : float, optional, default=2.0
        The exponential decay factor on the easy examples.
    neg_id : int, optional, default=0
        The negative id.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set ``normalization`` to *UNIT* will return the unreduced losses.

    """
    return Tensor.CreateOperator('SoftmaxFocalLoss', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def CTCLoss(
    inputs,
    blank_first=True,
    padding_mask=-1,
    use_softmax=True,
    **kwargs
):
    """Compute the ctc loss with batched variable length of labels. `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

    The data format of inputs should be *[T, N, C]*.

    If ``blank_first`` is *True*, *0* is reserved and

    label values are between *1* and *C - 1*.

    Otherwise, *C - 1* is reserved and *0* to *C - 2* can be used.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [logits, labels].
    blank_first : bool, optional
        Whether to put the blank at ``0``.
    padding_mask : int, optional
        The mask for padding the redundant labels.
    use_softmax : bool, optional
        Whether to use softmax before computing loss.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    The magnitude of loss is related to the ``sequence length``.

    """
    arguments = ParseArgs(locals())
    if use_softmax: arguments['inputs'][0] = \
        Softmax(arguments['inputs'][0], axis=2)
    return Tensor.CreateOperator('CTCLoss', **arguments)