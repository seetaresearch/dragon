# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor
import numpy as np

def SoftmaxLoss(inputs, axis=1, normalization='VALID', ignore_labels=(), **kwargs):
    """
    :param inputs:          a list of Tensor contains [input, label]
    :param axis             a int of using which axis to compute softmax
    :param normalization:   a str of (UNIT, FULL, VALID, BATCH_SIZE, NONE)
    :param ignore_labels:   a list of int contatins the labels to ignore
    :return:                a Tensor of loss with the shape (1,)
    """

    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('SoftmaxLoss Operator accpets a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='SoftmaxLoss', **kwargs)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        elif all(dim is not None for dim in inputs[0].shape):
            outer_dim = int(np.prod(inputs[0].shape[0 : axis]))
            inner_dim = int(np.prod(inputs[0].shape[axis + 1 :]))
            output.shape = [outer_dim * inner_dim]
        else: output.shape = [None]

    return output


def SigmoidCrossEntropyLoss(inputs, normalization='FULL', **kwargs):
    """
    :param inputs:          a list of Tensor contains [input, label]
    :param normalization:   a str of (UNIT, FULL, BATCH_SIZE, NONE)
    :return:                a Tensor of loss with the shape (1,)
    """

    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('SigmoidCrossEntropyLoss Operator accpets a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='SigmoidCrossEntropyLoss', **kwargs)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        else: output.shape = inputs[0].shape[:]

    return output


def SoftmaxCrossEntropyLoss(inputs, axis=1, normalization='FULL', **kwargs):
    """
    :param inputs:          a list of Tensor contains [input, label]
    :param normalization:   a str of (UNIT, FULL, BATCH_SIZE, NONE)
    :return:                a Tensor of loss with the shape (1,)
    """

    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('SoftmaxCrossEntropyLoss Operator accpets a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output =  Tensor.CreateOperator(nout=1, op_type='SoftmaxCrossEntropyLoss', **kwargs)

    if inputs[0].shape is not None:
        if normalization != 'UNIT': output.shape = [1]
        elif all(dim is not None for dim in inputs[0].shape):
            outer_dim = int(np.prod(inputs[0].shape[0 : axis]))
            inner_dim = int(np.prod(inputs[0].shape[axis + 1 :]))
            output.shape = [outer_dim * inner_dim]
        else: output.shape = [None]

    return output


def SmoothL1Loss(inputs, sigma=1.0, **kwargs):

    if not isinstance(inputs, list) or len(inputs) < 2:
        raise RuntimeError('SmoothL1Loss Operator accpets a list of at least 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='SmoothL1Loss', **kwargs)
    if inputs[0].shape is not None: output.shape = [1]
    return output


def L1Loss(inputs, normalization='BATCH_SIZE', coeff=1.0, **kwargs):

    if not isinstance(inputs, list) or len(inputs) < 2:
        raise RuntimeError('L1Loss Operator accpets a list of at least 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='L1Loss', **kwargs)
    if inputs[0].shape is not None: output.shape = [1]
    return output


def L2Loss(inputs, normalization='BATCH_SIZE', coeff=1.0, **kwargs):

    if not isinstance(inputs, list) or len(inputs) < 2:
        raise RuntimeError('L2Loss Operator accpets a list of at least 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='L2Loss', **kwargs)
    if inputs[0].shape is not None: output.shape = [1]
    return output