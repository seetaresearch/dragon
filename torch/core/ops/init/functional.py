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
"""Init functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core import cpp
from dragon.vm.torch.core.ops import utils
from dragon.vm.torch.core.ops.init import _functions


def arange(
    start,
    end=None,
    step=1,
    out=None,
    dtype='int64',
    device=None,
    requires_grad=False,
):
    """Return a tensor of evenly spaced values within a interval.

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
        The optional output tensor.
    dtype : str, optional, default='int64'
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        **True** to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        A vector with evenly spaced elements.

    """
    if end is None:
        slice_args = start, step
    else:
        slice_args = start, end, step
    out = _functions.Arange \
        .instantiate(
            device if device else cpp.device(),
            num_args=len(slice_args),
            dtype=dtype if dtype else 'int64',
        ).apply(slice_args, out)
    out.requires_grad = requires_grad
    return out


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
        The optional output tensor.
    dtype : str, optional, default='float32'
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        **True** to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    m = n if m is None else m
    out = utils.new_leaf([n, m], locals()) if out is None else out
    return _functions.Eye  \
        .instantiate(out.device, ndim=2, dtype=out.dtype) \
        .apply(out, [n, m])


def fill(out, shape, value):
    return _functions.Fill  \
        .instantiate(
            out.device,
            ndim=len(shape),
            value=float(value),
            dtype=out.dtype,
        ).apply(out, shape)


def fill_like(out, shape_like, value):
    return _functions.Fill \
        .instantiate(out.device, value=float(value), dtype=out.dtype) \
        .apply(out, [], shape_like)


def normal_fill(input, mean=0, std=1):
    """Fill input from the normal distribution."""
    shape = input.shape
    return _functions.RandomNormal \
        .instantiate(
            input.device,
            ndim=len(shape),
            mean=float(mean),
            std=float(std),
            dtype=input.dtype,
        ).apply(input, shape)


def ones(*size, **kwargs):
    r"""Return a tensor filled with ones.

    .. math:: \text{out} \leftarrow 1

    Parameters
    ----------
    size : int...
        The size(s) indicating the out shape.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.
    dtype : str, optional, default='float32'
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        **True** to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    out = kwargs.get('out', utils.new_leaf(size, kwargs))
    return fill(out, size, 1)


def ones_like(input, **kwargs):
    r"""Return a tensor of ones with shape as the other.

    .. math:: \text{out} \leftarrow 1

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The tensor for indicating shape.
    dtype : str, optional, default='float32'
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        **True** to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    out = utils.new_leaf(input.shape, kwargs)
    return fill_like(out, input, 1)


def rand(*size, **kwargs):
    """Return a tensor from the uniform distribution of U(0, 1).

    Parameters
    ----------
    size : int...
        The size(s) indicating the out shape.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.
    dtype : str, optional, default='float32'
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        **True** to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    out = kwargs.get('out', utils.new_leaf(size, kwargs))
    return uniform_fill(out, 0, 1)


def randn(*size, **kwargs):
    """Return a tensor from the normal distribution of N(0, 1).

    Parameters
    ----------
    size : int...
        The size(s) indicating the out shape.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.
    dtype : str, optional, default='float32'
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        **True** to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    out = kwargs.get('out', utils.new_leaf(size, kwargs))
    return normal_fill(out, 0, 1)


def uniform_fill(input, low=0, high=1):
    """Fill input from the uniform distribution."""
    shape = input.shape
    return _functions.RandomUniform \
        .instantiate(
            input.device,
            ndim=len(shape),
            low=float(low),
            high=float(high),
            dtype=input.dtype,
        ).apply(input, shape)


def zeros(*size, **kwargs):
    r"""Return a tensor filled with zeros.

    .. math:: \text{out} \leftarrow 0

    Parameters
    ----------
    size : int...
        The size(s) indicating the out shape.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.
    dtype : str, optional, default='float32'
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        **True** to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    out = kwargs.get('out', utils.new_leaf(size, kwargs))
    return fill(out, size, 0)


def zeros_like(input, **kwargs):
    r"""Return a tensor of zeros with shape as the other.

    .. math:: \text{out} \leftarrow 0

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The tensor for indicating shape.
    dtype : str, optional, default='float32'
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        **True** to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    out = utils.new_leaf(input.shape, kwargs)
    return fill_like(out, input, 0)
