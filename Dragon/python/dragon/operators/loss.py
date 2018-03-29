# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
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


def SmoothL1Loss(inputs, sigma=1.0, normalization='BATCH_SIZE', **kwargs):
    """SmoothL1Loss, introduced by `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, targets, inside_w, outside_w].
    sigma : float
        The sigma of L1 bound.
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


def SparseSoftmaxFocalLoss(inputs, axis=1, normalization='VALID', ignore_labels=(),
                           alpha=0.5, gamma=0.0, eps=1e-10, neg_id=-1, **kwargs):
    """SoftmaxFocalLoss with sparse labels, introduced by `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.

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
        The scale factor on the rare class. Default is ``0.5``.
    gamma : float
        The exponential decay factor on the easy examples. Default is ``0.0``.
    eps : float
        The eps.
    neg_id : int
        The negative id. Default is ``-1`` (Without Class Balance)

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

    output = Tensor.CreateOperator(nout=1, op_type='SparseSoftmaxFocalLoss', **arguments)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        elif all(dim is not None for dim in inputs[0].shape):
            outer_dim = int(np.prod(inputs[0].shape[0 : axis]))
            inner_dim = int(np.prod(inputs[0].shape[axis + 1 :]))
            output.shape = [outer_dim * inner_dim]
        else: output.shape = [None]

    return output