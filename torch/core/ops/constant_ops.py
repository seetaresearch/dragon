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

from dragon.core.framework import workspace
from dragon.core.util import nest
from dragon.vm.torch.core import cpp
from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.tensor import Tensor


def arange(
    start,
    end=None,
    step=1,
    out=None,
    dtype='int64',
    device=None,
    requires_grad=False,
):
    """Return a tensor of evenly spaced values within an interval.

    Specify ``start`` and ``end`` to determine an interval:

    ```python
    print(torch.arange(2, 4))  # [2, 3]
    ```

    If ``stop`` is **None**, interval :math:`[0, start)` will be taken instead:

    ```python
    print(torch.arange(5))  # [0, 1, 2, 3, 4]
    ```

    Set ``delta`` to make the strides:

    ```python
    print(torch.arange(5, step=2))  # [0, 2, 4]
    ```

    Parameters
    ----------
    start : number
        The start of interval.
    end : number, optional, default=0
        The stop of interval.
    step : number, optional, default=1
        The spacing between two elements.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='int64'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        A vector with evenly spaced elements.

    """
    if end is None:
        slice_args = start, step
    else:
        slice_args = start, end, step
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'Range', device, [], outputs=[out],
        dtype=dtype, num_args=len(slice_args), slice=slice_args)
    out._requires_grad = requires_grad
    return out


def empty(*size, dtype=None, device=None, requires_grad=False):
    """Return a tensor filled with uninitialized data.

    Parameters
    ----------
    size : int...
        The sizes of output tensor.
    dtype : str, optional
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device option.
    requires_grad : bool, optional, default=False
        Whether to compute the gradient if necessary.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Tensor(
        *size,
        dtype=dtype if dtype else 'float32',
        device=cpp.device() if device is None else device,
        requires_grad=requires_grad,
    )


def eye(
    n,
    m=None,
    out=None,
    dtype='float32',
    device=None,
    requires_grad=False,
):
    r"""Return a tensor constructed as the identity matrix.

    .. math:: \text{out} \leftarrow \text{diag}(1, 1, ..., 1)

    The rows and cols of matrix are determined by ``n`` and ``m``:

    ```python
    print(torch.eye(2))  # [[1., 0.], [0., 1.]]
    print(torch.eye(2, 3))  # [[1., 0., 0.], [0., 1., 0.]]
    ```

    Parameters
    ----------
    n : int
        The number output rows.
    m : int, optional
        The number output cols.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='float32'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    m = n if m is None else m
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'Eye', device, [], outputs=[out], dtype=dtype, ndim=2, dims=(n, m))
    out._requires_grad = requires_grad
    return out


def from_numpy(ndarray):
    """Create a tensor converting from the given numpy array.

    Parameters
    ----------
    ndarray : numpy.ndarray
        The numpy array data.

    Return
    ------
    dragon.vm.torch.Tensor
        The torch tensor.

    """
    if not isinstance(ndarray, numpy.ndarray):
        raise TypeError('<ndarray> should be a numpy array.')
    return Tensor(ndarray, copy=False)


def full(
    size,
    fill_value,
    out=None,
    dtype='int64',
    device=None,
    requires_grad=False,
):
    """Return a tensor filled with a scalar.

    Examples:

    ```python
    print(torch.full((1, 2), 1))  # [[1, 1]]
    ```

    Parameters
    ----------
    size : int...
        The output shape.
    fill_value : number
        The scalar to fill.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='int64'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    size = nest.flatten(size)
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'Fill', device, [], outputs=[out],
        dtype=dtype, value=float(fill_value), ndim=len(size), dims=size)
    out._requires_grad = requires_grad
    return out


def full_like(
    input,
    fill_value,
    out=None,
    dtype='int64',
    device=None,
    requires_grad=False,
):
    """Return a tensor filled with a scalar with size as input.

    Examples:

    ```python
    print(torch.full_like(torch.zeros(1, 2), 1))  # [[1, 1]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The tensor for indicating shape.
    fill_value : number
        The scalar to fill.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='int64'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'Fill', device, [input], dtype=dtype, value=float(fill_value))
    out._requires_grad = requires_grad
    return out


def linspace(
    start,
    end,
    steps=100,
    out=None,
    dtype='int64',
    dim=0,
    device=None,
    requires_grad=False,
):
    r"""Generate evenly spaced values within intervals along the given dimension.

    Interval :math:`[\text{start}, \text{end})` is determined for ``steps`` values:

    ```python
    x = torch.linspace(2, 4, steps=3)  # [2, 3, 4]
    ```

    More intervals are accepted to generate N-d coordinates:

    ```python
    x = torch.linspace([1, 2], [3, 4], steps=3, dim=0)  # [[1, 2], [2, 3], [3, 4]]
    y = torch.linspace([1, 2], [3, 4], steps=3, dim=1)  # [[1, 2, 3], [2, 3, 4]]
    ```

    Parameters
    ----------
    start : Union[number, Sequence[number]]
        The start(s) of interval.
    end: Union[number, Sequence[number]]
        The ends(s) of interval.
    steps : int, optional, default=100
        The number of values to generate.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='int64'
        The data type of output tensor.
    dim : int, optional, default=0
        The dimension to generate values.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    starts = nest.flatten(start)
    ends = nest.flatten(end)
    size = []
    if len(starts) > 1 or starts == start:
        size = [len(starts)]
    dim = dim if dim >= 0 else dim + len(size) + 1
    size.insert(dim, steps)
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'LinSpace', device, [], outputs=[out],
        dtype=dtype, axis=dim, ndim=len(size), num_intervals=len(starts),
        dims=size, start=starts, stop=ends)
    out._requires_grad = requires_grad
    return out


def ones(*size, out=None, dtype='float32', device=None, requires_grad=False):
    r"""Return a tensor filled with ones.

    .. math:: \text{out} \leftarrow 1

    Parameters
    ----------
    size : int...
        The output tensor shape.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='float32'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    size = nest.flatten(size)
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'Fill', device, [], outputs=[out],
        dtype=dtype, value=1.0, ndim=len(size), dims=size)
    out._requires_grad = requires_grad
    return out


def ones_like(input, dtype='float32', device=None, requires_grad=False):
    r"""Return a tensor of ones with shape as the other.

    .. math:: \text{out} \leftarrow 1

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The tensor for indicating the output shape.
    dtype : str, optional, default='float32'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    device = device or input.device
    out = Function.apply(
        'Fill', device, [input], dtype=dtype, value=1.0)
    out._requires_grad = requires_grad
    return out


def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create a tensor initializing from the given data.

    Parameters
    ----------
    data : array_like
        The data to initialize from.
    dtype : str, optional
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        ``True`` to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    array_data = numpy.array(data, copy=True)
    if dtype is None:
        dtype = str(array_data.dtype)
    else:
        array_data = array_data.astype(dtype)
    return Tensor(
        array_data,
        dtype=dtype,
        device=cpp.device() if device is None else device,
        requires_grad=requires_grad,
    )


def zeros(*size, out=None, dtype='float32', device=None, requires_grad=False):
    r"""Return a tensor filled with zeros.

    .. math:: \text{out} \leftarrow 0

    Parameters
    ----------
    size : int...
        The output tensor shape.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional, default='float32'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    size = nest.flatten(size)
    device = out.device if out else (device or cpp.device())
    out = Function.apply(
        'Fill', device, [], outputs=[out],
        dtype=dtype, value=0.0, ndim=len(size), dims=size)
    out._requires_grad = requires_grad
    return out


def zeros_like(input, dtype='float32', device=None, requires_grad=False):
    r"""Return a tensor of zeros with shape as the other.

    .. math:: \text{out} \leftarrow 0

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The tensor for indicating shape.
    dtype : str, optional, default='float32'
        The data type of output tensor.
    device : dragon.vm.torch.device, optional
        The device of output tensor.
    requires_grad : bool, optional, default=False
        Record gradient for output tensor or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    device = device or input.device
    out = Function.apply(
        'Fill', device, [input], dtype=dtype, value=0.0)
    out._requires_grad = requires_grad
    return out


def remove_scalars(input1, input2):
    """Remove the input scalars."""
    if isinstance(input1, Tensor):
        return input1, scalar(input2, input1.dtype, input1.device)
    return scalar(input1, input2.dtype, input2.device), input2


def scalar(input, dtype, device):
    """Return a cached scalar tensor.

    Parameters
    ----------
    input : number
        The scalar value.
    dtype : str, optional
        The data type of output tensor.
    device : dragon.vm.torch.device
        The device of output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if isinstance(input, Tensor):
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
    return Tensor(device=device, impl=impl)
