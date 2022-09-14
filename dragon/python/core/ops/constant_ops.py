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

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema
from dragon.core.framework import context as framework_context
from dragon.core.framework import types
from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor
from dragon.core.util import nest


def constant(
    value,
    dtype=None,
    shape=None,
    name=None,
    copy=True,
    symbolic=False,
):
    """Return a tensor initialized from the value.

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
            scope='Tensor' if symbolic
            else framework_context.get_variable_scope())
        .FromNumpy(initial_value, copy),
        deleter=None if symbolic else default_ws._handle_pool,
        symbolic=symbolic,
        name=name,
    )


def eye(n, m=None, k=0, dtype='float32', **kwargs):
    r"""Return a tensor constructed as the identity matrix.

    .. math:: \text{out} \leftarrow \text{diag}(1, 1, ..., 1)

    The rows and cols of matrix are determined by ``n`` and ``m``:

    ```python
    print(dragon.eye(2))     # [[1., 0.], [0., 1.]]
    print(dragon.eye(2, 3))  # [[1., 0., 0.], [0., 1., 0.]]
    ```

    The diagonal could be controlled by ``k``:

    * k > 0: Populate upper diagonal

    * k = 0: Populate main diagonal

    * k < 0: Populate lower diagonal

    Parameters
    ----------
    n : int
        The number of output rows.
    m : int, optional
        The number of output cols.
    k : int, optional, default=0
        The index of diagonal.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dims = (n, n if m is None else m)
    if context.executing_eagerly():
        return OpLib.execute('Eye', [], ndim=2, dims=dims, k=k, dtype=dtype)
    return OpLib.add('Eye', [], dims=dims, k=k, dtype=dtype, **kwargs)


@OpSchema.num_inputs(1)
def eye_like(inputs, k=0, dtype='float32', **kwargs):
    r"""Return a tensor of identity matrix with shape as the other.

    .. math:: \text{out} \leftarrow \text{diag}(1, 1, ..., 1)

    The rows and cols of matrix are hinted by the input tensor:

    ```python
    x = dragon.ones(2, 3)
    print(dragon.eye_like(x))  # [[1., 0.], [0., 1.]]
    ```

    The diagonal could be controlled by ``k``:

    * k > 0: Populate upper diagonal

    * k = 0: Populate main diagonal

    * k < 0: Populate lower diagonal

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor to hint the shape.
    k : int, optional, default=0
        The index of diagonal.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Eye', inputs, k=k, dtype=dtype)
    return OpLib.add('Eye', inputs, k=k, dtype=dtype, **kwargs)


@OpSchema.convert_arg(name='shape', name_v2='dims')
def fill(shape, value=0, dtype='float32', **kwargs):
    r"""Return a tensor filled with the scalar value.

    .. math:: \text{out} \leftarrow \text{value}

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    value : number, optional, default=0
        The value to fill.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    args['value'] = float(value)
    if context.executing_eagerly():
        return OpLib.execute('Fill', [], ndim=len(args['dims']), **args)
    return OpLib.add('Fill', [], **args)


def linspace(start, stop, num, dtype='int64', axis=0, **kwargs):
    r"""Generate evenly spaced values within intervals along the given axis.

    Range :math:`[\text{start}, \text{stop})` is determined for :attr:`num` values:

    ```python
    x = dragon.linspace(2, 4, num=3)  # [2, 3, 4]
    ```

    More ranges are accepted to generate N-d coordinates:

    ```python
    x = dragon.linspace([1, 2], [3, 4], num=3, axis=0)  # [[1, 2], [2, 3], [3, 4]]
    y = dragon.linspace([1, 2], [3, 4], num=3, axis=1)  # [[1, 2, 3], [2, 3, 4]]
    ```

    Parameters
    ----------
    start : Union[number, Sequence[number]]
        The start of range.
    stop: Union[number, Sequence[number]]
        The stop of range.
    num : int
        The number of values to generate.
    dtype : str, optional, default='int64'
        The optional data type.
    axis : int, optional, default=0
        The axis to generate values.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dtype = dtype.lower()
    starts = [float(elem) for elem in nest.flatten(start)]
    stops = [float(elem) for elem in nest.flatten(stop)]
    dims = []
    if len(starts) > 1 or starts == start:
        dims = [len(starts)]
    axis = axis if axis >= 0 else axis + len(dims) + 1
    dims.insert(axis, num)
    if context.executing_eagerly():
        return OpLib.execute(
            'LinSpace', [], ndim=len(dims), num_intervals=len(starts),
            axis=axis, dtype=dtype, dims=dims, start=starts, stop=stops)
    return OpLib.add('LinSpace', [], axis=axis, dtype=dtype, dims=dims,
                     start=starts, stop=stops, **kwargs)


@OpSchema.convert_arg(name='shape', name_v2='dims')
def ones(shape, dtype='float32', **kwargs):
    r"""Return a tensor filled with ones.

    .. math:: \text{out} \leftarrow 1

    ```python
    x = dragon.ones(shape=(2, 3), dtype='float32')
    ```

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return fill(shape, 1, dtype, **kwargs)


@OpSchema.num_inputs(1)
def ones_like(inputs, dtype='float32', **kwargs):
    r"""Return a tensor of ones with shape as the other.

    .. math:: \text{out} \leftarrow 1

    Examples:

    ```python
    x = dragon.ones(shape=(2, 3))
    y = dragon.ones_like(x)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor to hint the shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Fill', inputs, value=1.0, dtype=dtype)
    return OpLib.add('Fill', inputs, value=1.0, dtype=dtype, **kwargs)


def range(start, limit=None, delta=1, dtype='int64', **kwargs):
    """Return a tensor of evenly spaced values within an interval.

    Use :attr:`start` and :attr:`limit` for a simple range:

    ```python
    x = dragon.range(2, 4)  # [2, 3]
    ```

    If :attr:`limit` is ``None``, range :math:`[0, \text{start})` will be taken:

    ```python
    x = dragon.range(5)  # [0, 1, 2, 3, 4]
    ```

    Use :attr:`delta` to make a striding range:

    ```python
    x = dragon.range(5, delta=2)  # [0, 2, 4]
    ```

    Parameters
    ----------
    start : number
        The start of interval.
    limit: number, optional
        The end of interval.
    delta : number, optional, default=1
        The spacing between two elements.
    dtype : str, optional, default='int64'
        The output data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dtype = dtype.lower()
    slice = [float(start), float(delta)]
    if limit is not None:
        slice.insert(1, float(limit))
    if context.executing_eagerly():
        return OpLib.execute(
            'Range', [], num_args=len(slice), dtype=dtype, slice=slice)
    return OpLib.add('Range', [], slice=slice, dtype=dtype, **kwargs)


@OpSchema.convert_arg(name='shape', name_v2='dims')
def zeros(shape, dtype='float32', **kwargs):
    r"""Return a tensor filled with zeros.

    .. math:: \text{out} \leftarrow 0

    ```python
    x = dragon.zeros(shape=(2, 3), dtype='float32')
    ```

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return fill(shape, 0, dtype, **kwargs)


@OpSchema.num_inputs(1)
def zeros_like(inputs, dtype='float32', **kwargs):
    r"""Return a tensor of zeros with shape as the other.

    .. math:: \text{out} \leftarrow 0

    Examples:

    ```python
    x = dragon.zeros(shape=(2, 3))
    y = dragon.zeros_like(x)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor to hint the shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Fill', inputs, value=0.0, dtype=dtype)
    return OpLib.add('Fill', inputs, value=0.0, dtype=dtype, **kwargs)


def remove_scalars(inputs):
    """Remove the input scalars."""
    if len(inputs) == 2:
        if types.is_tensor(inputs[0]):
            inputs[1] = scalar(inputs[1], inputs[0].dtype)
        else:
            inputs[0] = scalar(inputs[0], inputs[1].dtype)
    return inputs


def scalar(input, dtype):
    """Return a tensor initialized from the scalar."""
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
