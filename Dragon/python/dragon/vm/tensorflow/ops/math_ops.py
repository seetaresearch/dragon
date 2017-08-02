# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = [
    'argmax',
    'matmul',
    'add',
    'subtract',
    'multiply',
    'divide',
    'sub',
    'mul',
    'div',
    'log',
    'exp',
    'square',
    'sqrt',
    'reduce_sum',
    'reduce_mean',
    'sigmoid',
    'tanh',
    'add_n'
]

from six.moves import range as xrange

import dragon.ops as ops


def argmax(input, axis=None, name=None, dimension=None):

    if dimension is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'dimension'.")
        axis = dimension
    elif axis is None: axis = 0
    return ops.Argmax(input, axis=axis, name=name)


def matmul(a,
           b,
           transpose_a=False,
           transpose_b=False,
           name=None):

    return ops.Matmul([a, b], TransA=transpose_a, TransB=transpose_b, name=name)


def add(x, y, name=None):

    return ops.Add([x, y], name=None)


def subtract(x, y, name=None):

    return ops.Sub([x, y], name=name)


def multiply(x, y, name=None):

    return ops.Mul([x, y], name=name)


def divide(x, y, name=None):

    return ops.Div([x, y], name=name)


def mul(x, y, name=None):

    return multiply(x, y, name)


def sub(x, y, name=None):

    return subtract(x, y, name)


def div(x, y, name=None):

    return divide(x, y, name=name)


def log(x, name=None):

    return ops.Log(x, name=name)


def exp(x, name=None):

    return ops.Exp(x, name=name)


def square(x, name=None):

    return ops.Square(x, name=name)


def sqrt(x, name=None):

    return ops.Pow(x, power=0.5, name=name)


def pow(x, power, name=None):

    return ops.Pow(x, power=power, name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):

    if reduction_indices is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'reduction_indices'.")
        axis = reduction_indices
    elif axis is None: axis = -1 # reduce all
    if isinstance(axis, list) or isinstance(axis, tuple): # reduce continuously
        if len(axis) < 1:
            raise RuntimeError('reduce axes should at least have one.')
        if len(axis) == 1:
            return ops.Sum(input_tensor, axis=axis[0], keep_dims=keep_dims)
        else:
            ret = ops.Sum(input_tensor, axis=axis[0], keep_dims=True)
            for i in xrange(1, len(axis) - 1):
                ret = ops.Sum(ret, axis=axis[i], keep_dims=True)
            return ops.Sum(ret, axis=axis[len(axis) - 1], keep_dims=keep_dims)
    else:
        return ops.Sum(input_tensor, axis=axis, keep_dims=keep_dims)


def reduce_mean(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):

    if reduction_indices is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'reduction_indices'.")
        axis = reduction_indices
    elif axis is None: axis = -1 # reduce all
    if isinstance(axis, list) or isinstance(axis, tuple): # reduce continuously
        if len(axis) < 1:
            raise RuntimeError('reduce axes should at least have one.')
        if len(axis) == 1:
            return ops.Mean(input_tensor, axis=axis[0], keep_dims=keep_dims)
        else:
            ret = ops.Mean(input_tensor, axis=axis[0], keep_dims=True)
            for i in xrange(1, len(axis) - 1):
                ret = ops.Mean(ret, axis=axis[i], keep_dims=True)
            return ops.Mean(ret, axis=axis[len(axis) - 1], keep_dims=keep_dims)
    else:
        return ops.Mean(input_tensor, axis=axis, keep_dims=keep_dims)


def sigmoid(x, name=None):

    return ops.Sigmoid(x, name=name)


def tanh(x, name=None):

    return ops.Tanh(x, name=name)


def add_n(inputs, name=None):

    return ops.Eltwise(inputs, operation='SUM', name=name)
