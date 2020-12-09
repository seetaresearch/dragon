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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.autograph.tensor import TensorRef
from dragon.core.eager import context as eager_context
from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import context
from dragon.core.framework import workspace


def constant(value, dtype=None, shape=None, name='Const'):
    """Return a tensor initialized from the value.

    Examples:

    ```python
    a = tf.constant(1)
    b = tf.constant(1, dtype='float32', shape=[1, 1, 1])
    c = tf.constant(numpy.ones((2, 3))
    ```

    Parameters
    ---------
    value : array_like
        The constant value.
    dtype : str, optional
        The optional data type.
    shape : Sequence[int], optional
        The optional tensor shape.
    name : str, optional
        The optional tensor name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dtype = str(dtype) if dtype else None
    if dtype is not None:
        if isinstance(value, numpy.ndarray):
            value = value.astype(dtype)
        else:
            value = numpy.array(value, dtype)
    else:
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value)
            # Discard the default 64bit types.
            if value.dtype == numpy.float64:
                value = value.astype(numpy.float32)
            elif value.dtype == numpy.int64:
                value = value.astype(numpy.int32)

    # Determine the shape.
    if shape is not None:
        if value.size == 1:
            # Case 1: Broadcast with scalar value.
            scalar = value.flatten()[0]
            value = numpy.empty(shape, value.dtype)
            value.fill(scalar)
        else:
            # Case 2: Reshape directly.
            value = value.reshape(shape)

    # Return a named tensor with value copied.
    name = context.get_name_scope() + name
    if eager_context.executing_eagerly():
        return EagerTensor(value, name=name + ':0')
    else:
        return TensorRef(
            name=workspace.get_workspace().unique_name(
                name, ':0', 'Tensor'),
            shape=list(value.shape),
            dtype=str(value.dtype),
        ).set_value(value)
