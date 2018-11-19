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

import numpy as np

from . import *
from .activation import Softmax


def NLLLoss(inputs, axis=1, normalization='VALID', ignore_labels=(), **kwargs):
    """Negative likelihood loss with sparse labels.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, sparse_labels].
    axis : int
        The axis of softmax function.
    normalization : str
        The normalization, ``UNIT``, ``FULL``, ``VALID``, ``BATCH_SIZE`` or ``NONE``.
    ignore_label : tuple or list
        The label id to ignore. Default is ``empty``.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set the normalization to ``UNIT`` will return unreduced losses.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='NLLLoss', **arguments)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        elif all(dim is not None for dim in inputs[0].shape):
            outer_dim = int(np.prod(inputs[0].shape[0 : axis]))
            inner_dim = int(np.prod(inputs[0].shape[axis + 1 :]))
            output.shape = [outer_dim * inner_dim]
        else: output.shape = [None]

    return output


def SparseSoftmaxCrossEntropy(inputs, axis=1, normalization='VALID', ignore_labels=(), **kwargs):
    """SoftmaxCrossEntropy with sparse labels.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, sparse_labels].
    axis : int
        The axis of softmax function.
    normalization : str
        The normalization, ``UNIT``, ``FULL``, ``VALID``, ``BATCH_SIZE`` or ``NONE``.
    ignore_label : tuple or list
        The label id to ignore. Default is ``empty``.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set the normalization to ``UNIT`` will return unreduced losses.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='SparseSoftmaxCrossEntropy', **arguments)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        elif all(dim is not None for dim in inputs[0].shape):
            outer_dim = int(np.prod(inputs[0].shape[0 : axis]))
            inner_dim = int(np.prod(inputs[0].shape[axis + 1 :]))
            output.shape = [outer_dim * inner_dim]
        else: output.shape = [None]

    return output


def SigmoidCrossEntropy(inputs, normalization='VALID', **kwargs):
    """SigmoidCrossEntropy with binary labels.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, labels].
    normalization : str
        The normalization, ``UNIT``, ``FULL``, ``VALID``, ``BATCH_SIZE`` or ``NONE``.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set the normalization to ``UNIT`` will return unreduced losses.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='SigmoidCrossEntropy', **arguments)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        else: output.shape = inputs[0].shape[:]

    return output


def SoftmaxCrossEntropy(inputs, axis=1, normalization='FULL', **kwargs):
    """SoftmaxCrossEntropy with one-hot labels.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, labels].
    axis : int
        The axis of softmax function.
    normalization : str
        The normalization, ``UNIT``, ``FULL``, ``BATCH_SIZE`` or ``NONE``.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set the normalization to ``UNIT`` will return unreduced losses.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output =  Tensor.CreateOperator(nout=1, op_type='SoftmaxCrossEntropy', **arguments)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        elif all(dim is not None for dim in inputs[0].shape):
            outer_dim = int(np.prod(inputs[0].shape[0 : axis]))
            inner_dim = int(np.prod(inputs[0].shape[axis + 1 :]))
            output.shape = [outer_dim * inner_dim]
        else: output.shape = [None]

    return output


def SmoothL1Loss(inputs, beta=1.0, normalization='BATCH_SIZE', **kwargs):
    """SmoothL1Loss. `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    Note that the ``beta`` is represented as ``1. / sigma / sigma`` following the original paper.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, targets, inside_w, outside_w].
    beta : float
        The transition point from L1 to L2 loss
    normalization : str
        The normalization, ``FULL``, ``BATCH_SIZE``, or ``NONE``.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    The number of inputs vary from ``2`` to ``4`` (Without or With ``inside_w/outside_w``).

    """
    CheckInputs(inputs, 2, 4)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='SmoothL1Loss', **arguments)
    if inputs[0].shape is not None: output.shape = [1]
    return output


def L1Loss(inputs, normalization='BATCH_SIZE', **kwargs):
    """L1Loss.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, targets, inside_w].
    normalization : str
        The normalization, ``FULL``, ``BATCH_SIZE``, or ``NONE``.

    Returns
    -------
    Tensor
        The loss, calculated as:  |l1_loss_function|

    Notes
    -----
    The number of inputs vary from ``2`` to ``3`` (Without or With ``inside_w``).

    """
    CheckInputs(inputs, 2, 3)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='L1Loss', **arguments)
    if inputs[0].shape is not None: output.shape = [1]
    return output


def L2Loss(inputs, normalization='BATCH_SIZE', **kwargs):
    """L2Loss(EuclideanLoss).

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, targets, inside_w].
    normalization : str
        The normalization, ``FULL``, ``BATCH_SIZE`` or ``NONE``.

    Returns
    -------
    Tensor
        The loss, calculated as:  |l2_loss_function|

    Notes
    -----
    The number of inputs vary from ``2`` to ``3`` (Without or With ``inside_w``).

    """
    CheckInputs(inputs, 2, 3)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='L2Loss', **arguments)
    if inputs[0].shape is not None: output.shape = [1]
    return output


def SigmoidFocalLoss(inputs, axis=1, normalization='VALID',
                     alpha=0.25, gamma=2.0, neg_id=0, **kwargs):
    """SigmoidFocalLoss with sparse labels. `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, sparse_labels].
    axis : int
        The axis of softmax function.
    normalization : str
        The normalization, ``UNIT``, ``FULL``, ``VALID``, ``BATCH_SIZE`` or ``NONE``.
    alpha : float
        The scale factor on the rare class. Default is ``0.25``.
    gamma : float
        The exponential decay factor on the easy examples. Default is ``2.0``.
    neg_id : int
        The negative id. Default is ``0``.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set the normalization to ``UNIT`` will return unreduced losses.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='SigmoidFocalLoss', **arguments)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        elif all(dim is not None for dim in inputs[0].shape):
            outer_dim = int(np.prod(inputs[0].shape[0 : axis]))
            inner_dim = int(np.prod(inputs[0].shape[axis + 1 :]))
            output.shape = [outer_dim * inner_dim]
        else: output.shape = [None]

    return output


def SoftmaxFocalLoss(inputs, axis=1, normalization='VALID', ignore_labels=(),
                     alpha=0.25, gamma=2.0, neg_id=0, **kwargs):
    """SoftmaxFocalLoss with sparse labels. `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, sparse_labels].
    axis : int
        The axis of softmax function.
    normalization : str
        The normalization, ``UNIT``, ``FULL``, ``VALID``, ``BATCH_SIZE`` or ``NONE``.
    ignore_label : tuple or list
        The label id to ignore. Default is ``empty``.
    alpha : float
        The scale factor on the rare class. Default is ``0.25``.
    gamma : float
        The exponential decay factor on the easy examples. Default is ``2.0``.
    neg_id : int
        The negative id. Default is ``0``.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    Set the normalization to ``UNIT`` will return unreduced losses.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='SoftmaxFocalLoss', **arguments)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        elif all(dim is not None for dim in inputs[0].shape):
            outer_dim = int(np.prod(inputs[0].shape[0 : axis]))
            inner_dim = int(np.prod(inputs[0].shape[axis + 1 :]))
            output.shape = [outer_dim * inner_dim]
        else: output.shape = [None]

    return output


def CTCLoss(inputs, blank_first=True, padding_mask=-1,
            use_softmax=True, **kwargs):
    """CTCLoss with batched variable length of labels. `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

    The data format of inputs should be ``[T, N, C]``.

    If ``blank_first`` is ``True``, ``0`` is reserved and

    label values are between ``1`` and ``C - 1``.

    Otherwise, ``C - 1`` is reserved and ``0`` to ``C - 2`` can be used.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, labels].
    blank_first : boolean
        Whether to put the blank at ``0``.
    padding_mask : int
        The mask for padding the redundant labels.
    use_softmax : boolean
        Whether to use softmax before computing loss.

    Returns
    -------
    Tensor
        The loss.

    Notes
    -----
    The magnitude of loss is related to the ``sequence length``.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())
    if use_softmax:
        arguments['inputs'][0] = Softmax(arguments['inputs'][0], axis=2)

    output = Tensor.CreateOperator(nout=1, op_type='CTCLoss', **arguments)
    if inputs[0].shape is not None: output.shape = [1]
    return output