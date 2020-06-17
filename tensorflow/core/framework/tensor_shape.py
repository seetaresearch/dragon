# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/tensor_shape.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph.tensor import Tensor
from dragon.core.eager.tensor import EagerTensor


class Dimension(object):
    def __init__(self, value):
        if value is None:
            self._value = None
        else:
            self._value = int(value)
            if self._value < 0:
                raise ValueError("Dimension %d must be >= 0" % self._value)

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return "Dimension(%s)" % repr(self._value)

    def __str__(self):
        value = self._value
        return "?" if value is None else str(value)

    def __eq__(self, other):
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return None
        return self._value == other.value

    def __ne__(self, other):
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return None
        return self._value != other.value

    def __int__(self):
        return self._value


def as_dimension(value):
    if isinstance(value, Dimension):
        return value
    else:
        return Dimension(value)


class TensorShape(object):
    def __init__(self, dims):
        if dims is None:
            self._dims = None
        elif isinstance(dims, TensorShape):
            self._dims = dims.dims
        else:
            try:
                dims_iter = iter(dims)
            except TypeError:
                self._dims = [as_dimension(dims)]
            else:
                self._dims = [as_dimension(d) for d in dims_iter]

    @property
    def dims(self):
        return self._dims

    @property
    def ndims(self):
        if self._dims is None:
            return None
        else:
            return len(self._dims)

    def as_list(self):
        if self._dims is None:
            raise ValueError("as_list() is not defined on an unknown TensorShape.")
        return [dim.value for dim in self._dims]

    def __repr__(self):
        return "TensorShape(%r)" % self._dims

    def __str__(self):
        if self.ndims is None:
            return "<unknown>"
        elif self.ndims == 1:
            return "(%s,)" % self._dims[0]
        else:
            return "(%s)" % ", ".join(str(d) for d in self._dims)

    def __getitem__(self, key):
        if self._dims is not None:
            if isinstance(key, slice):
                return TensorShape(self._dims[key])
            else:
                return self._dims[key]
        else:
            return Dimension(None)


def dimension_value(dimension):
    """Return the value of specified dimension."""
    if isinstance(dimension, Dimension):
        return dimension.value
    return dimension


def get_shape(self):
    """Construct the shape descriptor."""
    return TensorShape(self.shape)


# The Monkey Patching.
EagerTensor.get_shape = get_shape
Tensor.get_shape = get_shape
