# --------------------------------------------------------
# Theano @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor
import dragon.ops as ops

def batch_normalization(inputs, gamma, beta, mean, var, **kwargs):
    """Batch Normalization, introduced by `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    gamma: Tensor
        The scale parameter.
    beta: Tensor
        The shift parameter.
    mean: Tensor
        The moving average of mean.
    var: Tensor
        The moving average of variance.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return ops.BN([inputs, mean, var, gamma, beta])


def relu(x, alpha=0):
    """Rectified Linear Unit function, introduces by `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    alpha : float
        The slope of negative side.

    Returns
    -------
    Tensor
        The output tensor.

    """
    if alpha == 0: return ops.Relu(x)
    else: return ops.LRelu(x, slope=alpha)


def softmax(c):
    """Softmax function.

    The ``c`` should be a matrix, without any spatial axes.

    Parameters
    ----------
    c : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return ops.Softmax(c, axis=1)


def categorical_crossentropy(coding_dist, true_dist, axis=1):
    """Compute the categorical cross-entropy between input and target distribution.

    Parameters
    ----------
    coding_dist : Tensor
        The distribution of input.
    true_dist : Tensor
        The distribution of target.
    axis : int
        The axis of category.

    Returns
    -------
    Tensor
        The categorical cross-entropy.

    """
    return -ops.Sum(true_dist * ops.Log(coding_dist), axis=axis)


def sigmoid(x):
    """Sigmoid function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return ops.Sigmoid(x)


def tanh(x):
    """TanH function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return ops.Tanh(x)


def binary_crossentropy(output, target):
    """Compute the binary cross-entropy between input and target distribution.

    Parameters
    ----------
    output : Tensor
        The distribution of input.
    target : Tensor
        The distribution of target.

    Returns
    -------
    Tensor
        The binary cross-entropy.

    """
    return -(target * ops.Log(output) + (1.0 - target) * ops.Log(1.0 - output))





