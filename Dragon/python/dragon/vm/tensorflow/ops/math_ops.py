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

from dragon import ops as _ops


def argmax(input, axis=None, name=None, dimension=None):
    if dimension is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'dimension'.")
        axis = dimension
    elif axis is None: axis = 0
    return _ops.ArgMax(input, axis=axis, name=name)


def argmin(input, axis=None, name=None, dimension=None):
    if dimension is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'dimension'.")
        axis = dimension
    elif axis is None: axis = 0
    return _ops.ArgMin(input, axis=axis, name=name)


def matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    name=None,
):
    return _ops.Matmul(
        [a, b],
        transA=transpose_a,
        transB=transpose_b,
        name=name,
    )


def add(x, y, name=None):
    return _ops.Add([x, y], name=name)


def subtract(x, y, name=None):
    return _ops.Sub([x, y], name=name)


def multiply(x, y, name=None):
    return _ops.Mul([x, y], name=name)


def divide(x, y, name=None):
    return _ops.Div([x, y], name=name)


def mul(x, y, name=None):
    return multiply(x, y, name)


def sub(x, y, name=None):
    return subtract(x, y, name)


def div(x, y, name=None):
    return divide(x, y, name=name)


def cast(x, dtype, name=None):
    return _ops.Cast(x, dtype=dtype, name=name)


def log(x, name=None):
    return _ops.Log(x, name=name)


def exp(x, name=None):
    return _ops.Exp(x, name=name)


def square(x, name=None):
    return _ops.Square(x, name=name)


def sqrt(x, name=None):
    return _ops.Pow(x, power=0.5, name=name)


def pow(x, power, name=None):
    return _ops.Pow(x, power=power, name=name)


def reduce_sum(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None,
):
    if reduction_indices is not None:
        if axis is not None:
            raise ValueError(
                "Cannot specify both 'axis' and 'reduction_indices'.")
        axis = reduction_indices
    return _ops.Sum(
        input_tensor,
        axes=axis,
        keep_dims=keep_dims,
        name=name,
    )


def reduce_mean(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None,
):
    if reduction_indices is not None:
        if axis is not None:
            raise ValueError(
                "cannot specify both 'axis' and 'reduction_indices'.")
        axis = reduction_indices
    return _ops.Mean(
        input_tensor,
        axes=axis,
        keep_dims=keep_dims,
        name=name,
    )


def sigmoid(x, name=None):
    return _ops.Sigmoid(x, name=name)


def tanh(x, name=None):
    return _ops.Tanh(x, name=name)


def add_n(inputs, name=None):
    return _ops.Eltwise(inputs, operation='SUM', name=name)
