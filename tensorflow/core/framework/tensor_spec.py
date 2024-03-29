# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Tensor spec utilities."""

from dragon.vm.tensorflow.core.framework import dtypes
from dragon.vm.tensorflow.core.framework import tensor_shape


class TensorSpec(object):
    """Spec to describe properties of a tensor."""

    def __init__(self, shape, dtype="float32", name=None):
        """Create a TensorSpec.

        Parameters
        ----------
        shape : Sequence[int], required
            The dimensions.
        dtype : str, optional, default='float32'
            The optional data type.
        name : str, optional
            The optional name.

        """
        self._shape = tensor_shape.TensorShape(shape)
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
        return str(self._dtype)

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

    def is_compatible_with(self, spec_or_tensor):
        """Return a bool whether given the spec is compatible.

        Returns
        -------
        bool
            ``True`` if compatible otherwise ``False``.

        """

        def dtype_is_compatible_with(other):
            return self.dtype == other.dtype

        def shape_is_compatible_with(other):
            shape = other.shape
            if self._shape is not None and shape is not None:
                if len(self.shape) != len(shape):
                    return False
            for x_dim, y_dim in zip(self.shape, shape):
                if x_dim != y_dim:
                    return False
            return True

        return dtype_is_compatible_with(spec_or_tensor) and shape_is_compatible_with(spec_or_tensor)
