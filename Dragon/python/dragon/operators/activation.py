# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from . import *

def Relu(inputs, **kwargs):
    """Rectified Linear Unit function, introduces by `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |relu_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Relu', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def LRelu(inputs, slope=0.2, **kwargs):
    """Leaky Rectified Linear Unit function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    slope : float
        The slope of negative side.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |leaky_relu_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Relu', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Elu(inputs, alpha=1.0, **kwargs):
    """Exponential Linear Unit function, introduces by `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |elu_function|

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Elu', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Sigmoid(inputs, **kwargs):
    """Sigmoid function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |sigmoid_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Sigmoid', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Tanh(inputs, **kwargs):
    """Tanh function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |tanh_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Tanh', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Dropout(inputs, prob=0.5, scale=True, **kwargs):
    """Randomly set a unit into zero, introduced by `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    prob : float
        The prob of dropping. Default is ``0.5``.
    scale : boolean
        Whether to scale the output during training.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |dropout_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Dropout', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Softmax(inputs, axis=1, **kwargs):
    """Softmax function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The axis to perform softmax.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |softmax_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Softmax', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output