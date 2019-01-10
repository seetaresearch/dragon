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

import dragon 


__all__ = [
    'argmax',
    'argmin',
    'matmul',
    'add',
    'subtract',
    'multiply',
    'divide',
    'sub',
    'mul',
    'div',
    'cast',
    'log',
    'exp',
    'square',
    'sqrt',
    'reduce_sum',
    'reduce_mean',
    'sigmoid',
    'tanh',
    'add_n',
]


def argmax(input, axis=None, name=None, dimension=None):
    if dimension is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'dimension'.")
        axis = dimension
    elif axis is None: axis = 0
    return dragon.ops.ArgMax(input, axis=axis, name=name)


def argmin(input, axis=None, name=None, dimension=None):
    if dimension is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'dimension'.")
        axis = dimension
    elif axis is None: axis = 0
    return dragon.ops.ArgMin(input, axis=axis, name=name)


def matmul(a,
           b,
           transpose_a=False,
           transpose_b=False,
           name=None):
    return dragon.ops.Matmul([a, b], transA=transpose_a, transB=transpose_b, name=name)


def add(x, y, name=None):
    return dragon.ops.Add([x, y], name=name)


def subtract(x, y, name=None):
    return dragon.ops.Sub([x, y], name=name)


def multiply(x, y, name=None):
    return dragon.ops.Mul([x, y], name=name)


def divide(x, y, name=None):
    return dragon.ops.Div([x, y], name=name)


def mul(x, y, name=None):
    return multiply(x, y, name)


def sub(x, y, name=None):
    return subtract(x, y, name)


def div(x, y, name=None):
    return divide(x, y, name=name)


def cast(x, dtype, name=None):
    return dragon.ops.Cast(x, dtype=dtype, name=name)


def log(x, name=None):
    return dragon.ops.Log(x, name=name)


def exp(x, name=None):
    return dragon.ops.Exp(x, name=name)


def square(x, name=None):
    return dragon.ops.Square(x, name=name)


def sqrt(x, name=None):
    return dragon.ops.Pow(x, power=0.5, name=name)


def pow(x, power, name=None):
    return dragon.ops.Pow(x, power=power, name=name)


def reduce_sum(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None
):
    if reduction_indices is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'reduction_indices'.")
        axis = reduction_indices
    return dragon.ops.Sum(input_tensor, axes=axis, keep_dims=keep_dims, nama=name)


def reduce_mean(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None
):
    if reduction_indices is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'reduction_indices'.")
        axis = reduction_indices
    return dragon.ops.Mean(input_tensor, axes=axis, keep_dims=keep_dims, nama=name)


def sigmoid(x, name=None):
    return dragon.ops.Sigmoid(x, name=name)


def tanh(x, name=None):
    return dragon.ops.Tanh(x, name=name)


def add_n(inputs, name=None):
    return dragon.ops.Eltwise(inputs, operation='SUM', name=name)
