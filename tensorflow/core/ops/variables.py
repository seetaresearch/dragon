# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Variable class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor


class Variable(Tensor):
    """Resource variable."""

    def __init__(
        self,
        initial_value,
        trainable=True,
        name=None,
        dtype=None,
        shape=None,
    ):
        """Create a ``Variable``."""
        # Determine the initial value.
        if isinstance(initial_value, Tensor):
            value = initial_value.numpy()
        else:
            value = initial_value
        # Determine the data type and shape.
        dtype = str(dtype) if dtype is not None else dtype
        value = numpy.array(value, dtype, copy=False)
        if shape is not None:
            if value.size == 1:
                # Broadcast with scalar value.
                scalar = value.flatten()[0]
                value = numpy.empty(shape, value.dtype)
                value.fill(scalar)
            else:
                # Reshape.
                value = value.reshape(shape)
        # Initialize tensor from the value.
        default_ws = workspace.get_workspace()
        super(Variable, self).__init__(
            shape=value.shape,
            dtype=value.dtype,
            impl=default_ws.create_tensor(
                scope=context.get_variable_scope())
            .FromNumpy(value, True),
            deleter=default_ws._handle_pool,
            name=name,
        )
        self.requires_grad = trainable

    @property
    def trainable(self):
        """Return a bool indicating if this variable is trainable.

        Returns
        -------
        bool
            **True** if trainable otherwise **False**.

        """
        return self._requires_grad

    def __repr__(self):
        array = self.numpy()
        content_str, shape = str(array), array.shape
        numpy_str = '{}, dtype={}'.format(content_str, array.dtype)
        del array  # DECREF
        if len(shape) == 0:
            return content_str
        shape_str = ('(' + ', '.join(
            [str(dim) for dim in shape])) + \
            (',)' if len(shape) == 1 else ')')
        return '<tf.Variable {} shape={} dtype={}, numpy=\n{}>' \
            .format(self.name, shape_str, self.dtype, numpy_str)
