# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Structure to represent a tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest


class TensorSpec(object):
    """Spec to describe properties of a tensor."""

    def __init__(self, shape, dtype='float32', name=None):
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
        self._shape, self._dtype, self._name = shape, dtype, name
        if shape is not None:
            self._shape = nest.flatten(shape)

    @property
    def dtype(self):
        """Return the data type.

        Returns
        -------
        str
            The data type.

        """
        return self._dtype

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
        return self._shape

    def is_compatible_with(self, spec_or_tensor):
        """Return a bool indicating whether given the spec is compatible.

        Returns
        -------
        bool
            **True** if ``shape`` and ``dtype`` are
            both compatible otherwise **False**.

        """
        def dtype_is_compatible_with(spec_or_tensor):
            return self._dtype == spec_or_tensor.dtype

        def shape_is_compatible_with(spec_or_tensor):
            shape = spec_or_tensor.shape
            if self._shape is not None and shape is not None:
                if len(self._shape) != len(shape):
                    return False
            for x_dim, y_dim in zip(self._shape, shape):
                if x_dim != y_dim:
                    return False
            return True

        return \
            dtype_is_compatible_with(spec_or_tensor) and \
            shape_is_compatible_with(spec_or_tensor)
