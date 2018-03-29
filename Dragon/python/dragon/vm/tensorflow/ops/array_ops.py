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

__all__ = [
    'expand_dims',
    'shape',
    'zeros',
    'ones',
    'placeholder',
    'concat',
    'transpose',
    'tile',
    'reshape'
]


import dragon.ops as ops
from dragon.core.tensor import Tensor

from dragon.vm.tensorflow.framework import dtypes


def expand_dims(input, axis=None, name=None, dim=None):

    if dim is not None:
        if axis is not None:
            raise ValueError("cannot specify both 'axis' and 'dim'.")
        axis = dim

    return ops.ExpandDims(input, axis=axis, name=name)


def shape(input, name=None, out_type=dtypes.float32):

    return ops.Shape(input, name=None)


def zeros(shape, dtype=dtypes.float32, name=None):

    return ops.Fill(shape, value=0.0, name=name)


def ones(shape, dtype=dtypes.float32, name=None):

    return ops.Fill(shape, value=1.0, name=name)


def placeholder(dtype, shape=None, name=None):
    # check data type
    if dtype is not None:
        if not isinstance(dtype, dtypes.DType):
            raise TypeError('The dtype should be a valid tf data type.')
        dtype = dtype.name

    return Tensor(name=name, shape=shape, dtype=dtype).Placeholder()


def concat(values, axis, name=None):

    return ops.Concat(values, axis=axis, name=name)


def transpose(a, perm=None, name=None):

    return ops.Transpose(a, perm=perm, name=name)


def tile(input, multiples, name=None):

    return ops.Tile(input, multiples=multiples, name=name)


def reshape(tensor, shape, name=None):

    return ops.Reshape(tensor, shape=shape, name=None)
