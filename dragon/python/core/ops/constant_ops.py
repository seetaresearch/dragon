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
"""Constant ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.framework import context
from dragon.core.framework import types
from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor


def constant(
    value,
    dtype=None,
    shape=None,
    name=None,
    copy=True,
    symbolic=False,
):
    r"""Return a tensor initialized from the value.

    Examples:

    ```python
    a = dragon.constant(1)
    b = dragon.constant(1, dtype='float32', shape=(1, 1, 1))
    c = dragon.constant(numpy.ones((2, 3))
    ```

    Parameters
    ----------
    value : array_like
        The value to initialize from.
    dtype : str, optional
        The optional data type.
    shape : Sequence[int], optional
        The optional tensor shape.
    name : str, optional
        The optional tensor name.
    copy : bool, optional, default=True
        Whether to copy the value.
    symbolic : bool, optional, default=False
        Whether to initialize as a symbolic tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    # Determine the initial value.
    if isinstance(value, Tensor):
        initial_value = value.numpy()
    else:
        initial_value = value
    # Determine the data type and shape.
    initial_value = numpy.array(initial_value, dtype, copy=False)
    if shape is not None:
        if initial_value.size == 1:
            # Broadcast with scalar value.
            scalar = initial_value.flatten()[0]
            initial_value = numpy.empty(shape, initial_value.dtype)
            initial_value.fill(scalar)
        else:
            # Reshape.
            initial_value = initial_value.reshape(shape)
    # Return a tensor initialized from the value.
    default_ws = workspace.get_workspace()
    return Tensor(
        shape=initial_value.shape,
        dtype=initial_value.dtype,
        impl=default_ws.create_tensor(
            scope='Symbol' if symbolic
            else context.get_variable_scope())
        .FromNumpy(initial_value, copy),
        deleter=None if symbolic else default_ws._handle_pool,
        symbolic=symbolic,
        name=name,
    )


def remove_scalars(inputs):
    """Remove the input scalars."""
    if len(inputs) == 2:
        if types.is_tensor(inputs[0]):
            inputs[1] = get_scalar(inputs[1], inputs[0].dtype)
        else:
            inputs[0] = get_scalar(inputs[0], inputs[1].dtype)
    return inputs


def get_scalar(input, dtype):
    """Return a cached scalar."""
    if types.is_tensor(input):
        return input
    try:
        input = float(input)
    except (TypeError, ValueError):
        raise ValueError(
            '<input> should be a python number, got {}.'
            .format(type(input).__name__))
    cached_name = '%s(%s)' % (dtype, input)
    default_ws = workspace.get_workspace()
    impl = default_ws.get_tensor(cached_name)
    if impl is None:
        impl = default_ws.create_tensor(cached_name)
        impl.FromNumpy(numpy.array(input, dtype), True)
    return Tensor((), dtype, impl=impl, symbolic=True)
