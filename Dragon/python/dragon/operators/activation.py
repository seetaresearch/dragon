# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor

def Relu(inputs, **kwargs):
    """

    :param inputs:      a Tensor with any shape
    :return:            a Tensor of { max(x,0) }

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Relu Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Relu', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def LRelu(inputs, slope=0.2, **kwargs):
    """

    :param inputs:      a Tensor with any shape
    :param slope:       a float of the slope
    :return:            a Tensor of { max(x,0) + slope * min(x, 0) )

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Relu Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Relu', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Sigmoid(inputs, **kwargs):
    """

    :param inputs:      a Tensor with any shape
    :return:            a Tensor of { 1 / (1 + e^{-x}) }

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Sigmoid Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Sigmoid', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Tanh(inputs, **kwargs):
    """

    :param inputs:      a Tensor with any shape
    :return:            a Tensor of { tanh(x) } Tensor

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Tanh Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Tanh', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Dropout(inputs, prob, scale=True, **kwargs):
    """

    :param inputs:      a Tensor with any shape
    :return:            a Tensor of { zero~prob(x) }

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Dropout Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Dropout', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Softmax(inputs, axis=1, **kwargs):
    """

    :param inputs:      a Tensor with any shape
    :return:            a Tensor { e^(xi) / \sigma e^(xj) }

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Softmax Operator accepts a Tensor as inputs')

    args = locals(); wargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Softmax', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output