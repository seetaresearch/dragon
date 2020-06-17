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
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/tensor_spec.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import tensor_spec
from dragon.vm.tensorflow.core.framework import dtypes
from dragon.vm.tensorflow.core.framework import tensor_shape


class TensorSpec(tensor_spec.TensorSpec):
    """Spec to describe properties of a tensor."""

    def __init__(self, shape, dtype=dtypes.float32, name=None):
        """Create a ``TensorSpec``."""
        self._shape = tensor_shape.TensorShape(shape)
        try:
            self._shape_tuple = tuple(self._shape.as_list())
        except ValueError:
            self._shape_tuple = None
        self._dtype = dtypes.as_dtype(dtype)
        self._name = name

    @property
    def dtype(self):
        """Return the data type.

        Returns
        -------
        str
            The data type.

        """
        return self._dtype.name

    @property
    def name(self):
        """Return the spec name.

        Returns
        -------
        str
            The spec name.

        """
        return self._name

    @property
    def shape(self):
        """Return the dimensions.

        Returns
        -------
        Sequence[int]
            The dimensions.

        """
        return self._shape.as_list()
