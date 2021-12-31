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
"""Array ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest
from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.ops import constant_ops
from dragon.vm.torch.core.tensor import Tensor


def cat(tensors, dim=0, out=None):
    """Concatenate the inputs along the given dimension.

    Parameters
    ----------
    tensors : Sequence[dragon.vm.torch.Tensor]
        The input tensors.
    dim : int, optional
        The dimension to concatenate.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Concat', tensors[0].device, tensors, outputs=[out], axis=dim)


def chunk(tensor, chunks, dim=0, copy=True):
    """Split input into a specific number of chunks.

    Examples:

    ```python
    x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # Shape: (3, 2) -> (2, 2), (1, 2)
    print(torch.chunk(x, chunks=2))
    ```

    :attr:`dim` could be negative:

    ```python
    x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(torch.chunk(x, 2, dim=1))
    print(torch.chunk(x, 2, dim=-1))  # Equivalent
    ```

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    chunks : int
        The number of chunks to split.
    dim : int, optional, default=0
        The dimension to split.
    copy : bool, optional, default=True
        Copy or create the views of input.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The output tensors.

    """
    return Function.apply(
        'Split', tensor.device, [tensor], outputs=[None] * chunks,
        axis=dim, num_splits=0, copy=copy)


def broadcast_to(input, shape):
    """Broadcast input to the given shape.

    Length of ``shape`` could either be less or more
    than the number of input dimensions:

    ```python
    a = torch.tensor([[1], [2], [3]])
    # Shape: (3, 1) -> (3, 2)
    print(torch.broadcast_to(a, shape=(3, 2)))
    print(torch.broadcast_to(a, shape=(2,)))  # Equivalent

    # Shape: (3,) -> (1, 3) -> (2, 3)
    b = torch.tensor([1, 2, 3])
    print(torch.broadcast_to(b, shape=(2, 3)))

    # Wrong remapping shape: (3,) -> (6,)
    # Only the dimension with size 1 could broadcast
    print(torch.broadcast_to(b, shape=(6,)))
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    shape : Sequence[int]
        The output shape.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Expand', input.device, [input], ndim=len(shape), dims=shape)


def flatten(input, start_dim=0, end_dim=-1, out=None):
    """Return a tensor with dimensions flattened.

    :attr:`start_dim` and :attr:`end_dim` could be negative:

    ```python
    # A negative dimension is the last-k dimension
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(torch.flatten(x, start_dim=0, end_dim=1))
    print(torch.flatten(x, start_dim=0, end_dim=-1))  # Equivalent
    ```

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    start_dim : int, optional, default=0
        The start dimension to flatten.
    end_dim : int, optional, default=-1
        The end dimension to flatten.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Flatten', input.device, [input], outputs=[out],
        axis=start_dim, end_axis=end_dim)


def flip(input, dims):
    """Reverse elements along the given dimension.

    :attr:`dims` could be negative:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative dimension is the last-k dimension
    print(torch.flip(x, dims=1))  # [[3, 2, 1], [6, 5, 4]]
    print(torch.flip(x, dims=-1))  # Equivalent

    # Also, dimension could be a sequence of integers
    print(torch.flip(x, dims=(0, 1)))  # [[6, 5, 4], [3, 2, 1]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dims : Union[int, Sequence[int]]
        The dimension to reverse.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Reverse', input.device, [input],
        axes=nest.flatten(dims) if dims is not None else dims)


def fliplr(input):
    """Reverse elements along the second dimension.

    Examples:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(torch.fliplr(x))  # [[3, 2, 1], [6, 5, 4]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return flip(input, 1)


def flipud(input):
    """Reverse elements along the first dimension.

    Examples:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(torch.flipud(x))  # [4, 5, 6], [1, 2, 3]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return flip(input, 0)


def gather(input, dim, index, out=None):
    """Gather elements along the given dimension of index.

    Number of dimensions of :attr:`input`, :attr:`index` should be same.
    For 3-d input, output is gathered as:

    ```python
    out[i, j, k] = input[index[i, j, k], j, k]
    out[i, j, k] = input[i, index[i, j, k], k]
    out[i, j, k] = input[i, j, index[i, j, k]]
    ```

    Examples:

    ```python
    x = torch.tensor([[1, 2], [3, 4]])
    index = torch.tensor([[0, 0], [0, 1]])
    print(torch.gather(x, 0, index))  # [[1, 2], [1, 4]]
    print(torch.gather(x, 1, index))  # [[1, 1], [3, 4]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int
        The dimension of index values.
    index : dragon.vm.torch.Tensor
        The index tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    """
    return Function.apply(
        'GatherElements', input.device, [input, index],
        outputs=[out], axis=dim)


def index_select(input, dim, index, out=None):
    """Select the elements along the given dimension using index.

    Index should be a ``int64`` tensor:

    ```python
    input = torch.tensor([[1, 2, 3], [4, 5, 6]])
    index = torch.tensor([1])
    print(torch.index_select(input, 0, index))  # [[4, 5, 6]]
    ```

    Continuous dimensions could also be specified to index:

    ```python
    print(torch.index_select(input, (0, 1), index))  # [2]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]]
        The dimension to select.
    index : dragon.vm.torch.Tensor
        The index tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    ndim = input.ndimension()
    dims = [v if v >= 0 else v + ndim for v in nest.flatten(dim)]
    dims.sort()
    if dims[-1] != (dims[0] + len(dims) - 1):
        raise ValueError('<dim> should be a continuous sequence.')
    return Function.apply(
        'Gather', input.device, [input, index], outputs=[out],
        axis=dims[0], end_axis=dims[-1])


def masked_fill(input, mask, value, out=None):
    """Fill tensor with the value where mask is true.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    mask : dragon.vm.torch.Tensor
        The boolean mask.
    value : Union[number, dragon.vm.torch.Tensor]
        The value to fill.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if not isinstance(value, Tensor):
        value = constant_ops.scalar(value, input.dtype, input.device)
    return Function.apply(
        'Where', input.device, [mask, value, input], outputs=[out])


def masked_select(input, mask, out=None):
    """Select the input elements where mask is true.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    mask : dragon.vm.torch.Tensor
        The mask for selecting.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'BooleanMask', input.device, [input, mask], outputs=[out])


def multinomial(input, num_samples, out=None):
    """Return an index tensor sampled from the multinomial distribution.

    Examples:

    ```python
    input = torch.tensor([0.5, 0.5]).log()
    index = torch.multinomial(input, 1)
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    num_samples : int
        The number of samples in each row.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Multinomial', input.device, [input], outputs=[out],
        sample_size=num_samples)


def narrow(input, dimension, start, length):
    """Return a narrowed tensor of input.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    dimension : int
        The dimension to slice.
    start : int
        The starting position.
    length : int
        The distance to the ending position.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    sizes = list(input.shape[:])
    starts = [0] * len(sizes)
    starts[dimension], sizes[dimension] = start, length
    return Function.apply(
        'Slice', input.device, [input], ndim=len(starts),
        starts=starts, sizes=sizes)


def nonzero(input, out=None):
    r"""Return the index of non-zero elements.

    .. math:: \text{out} = \{i\}, \text{ if } \text{input}_{i} \neq 0

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply('NonZero', input.device, [input], outputs=[out])


def permute(input, dims, out=None):
    """Return a tensor with the new order of dimensions.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dims : Sequence[int]
        The output of dimensions.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Transpose', input.device, [input], outputs=[out],
        ndim=len(dims), perm=dims)


def tile(input, reps):
    """Repeat elements along each dimension of input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    reps : Sequence[int]
        The repetition for each dimension.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Tile', input.device, [input], ndim=len(reps), repeats=reps)


def reshape(input, shape, out=None):
    """Change the shape of input.

    Examples:

    ```python
    # Provide a determined value for each dimension if possible
    x = torch.ones(1, 2, 3, 4)
    print(torch.reshape(x, shape=(6, 4)).shape)  # (6, 4)

    # Set the existing dimensions to ``0`` if it unchanged
    print(torch.reshape(x, shape=(0, 0, 12)).shape)  # (1, 2, 12)
    print(torch.reshape(x, shape=(0, 0, 0, 0)).shape)  # (1, 2, 3, 4)
    print(torch.reshape(x, shape=(0, 0, 0, 0, 0)).shape)  # Wrong

    # You can also set ``-1`` once to infer the value
    print(torch.reshape(x, shape=(-1, 4)).shape)  # (6, 4)
    print(torch.reshape(x, shape=(-1, -1)).shape)  # Wrong
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    shape : Sequence[int]
        The output shape.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Reshape', input.device, [input], outputs=[out],
        ndim=len(shape), dims=shape)


def roll(input, shifts, dims=None):
    """Roll elements along the given dimension.

    :attr:`dims` could be negative or ``None``:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative dimension is the last-k dimension
    print(torch.roll(x, shifts=1, dims=1))  # [[3, 1, 2], [6, 4, 5]]
    print(torch.roll(x, shifts=1, dims=-1))  # Equivalent

    # If dimension is None, roll input as a vector
    print(torch.roll(x, shifts=1))  # [[6, 1, 2], [3, 4, 5]]

    # Also, dimension could be a sequence of integers
    print(torch.roll(x, shifts=(1, 1), dims=(0, 1)))  # [[6, 4, 5], [3, 1, 2]]
    print(torch.roll(x, shifts=(1, -1), dims=(0, 1)))  # [[5, 6, 4], [2, 3, 1]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    shifts : Union[int, Sequence[int]]
        The rolling offset of each dimension.
    dims : Union[int, Sequence[int]], optional
        The dimension to roll.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    shifts = nest.flatten(shifts)
    dims = nest.flatten(dims) if dims is not None else dims
    return Function.apply(
        'Roll', input.device, [input],
        num_shifts=len(shifts), shifts=shifts, axes=dims)


def scatter(input, dim, index, src, out=None):
    """Update elements along the given dimension of index.

    Number of dimensions of :attr:`input`, :attr:`index`, and :attr:`src`
    should be same. For 3-d input, output is updated as:

    ```python
    out[index[i, j, k], j, k] = src[i, j, k]  # ``dim`` is 0
    out[i, index[i, j, k], k] = src[i, j, k]  # ``dim`` is 1
    out[i, j, index[i, j, k]] = src[i, j, k]  # ``dim`` is 2
    ```

    Examples:

    ```python
    y = torch.tensor([[1, 2], [3, 4]])
    x = torch.tensor([[5, 6], [7, 8]])
    index = torch.tensor([[0, 1], [1, 0]])
    print(torch.scatter(y, 0, index, x))  # [[5, 8], [7, 6]]
    print(torch.scatter(y, 1, index, x))  # [[5, 6], [8, 7]]
    print(torch.scatter(y, 0, index, 8))  # [[8, 8], [8, 8]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int
        The dimension of index values.
    index : dragon.vm.torch.Tensor
        The index tensor.
    src : Union[dragon.vm.torch.Tensor, number]
        The tensor to update from.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    """
    if not isinstance(src, Tensor):
        src = constant_ops.full_like(
            index, src, dtype=input.dtype, device=input.device)
    return Function.apply(
        'ScatterElements', input.device, [input, index, src],
        outputs=[out], axis=dim)


def scatter_add(input, dim, index, src, out=None):
    """Add elements along the given dimension of index.

    Number of dimensions of :attr:`input`, :attr:`index`, and :attr:`src`
    should be same. For 3-d input, output is updated as:

    ```python
    out[index[i, j, k], j, k] += src[i, j, k]  # ``dim`` is 0
    out[i, index[i, j, k], k] += src[i, j, k]  # ``dim`` is 1
    out[i, j, index[i, j, k]] += src[i, j, k]  # ``dim`` is 2
    ```

    Examples:

    ```python
    y = torch.tensor([[1, 2], [3, 4]])
    x = torch.tensor([[5, 6], [7, 8]])
    index = torch.tensor([[0, 0], [0, 0]])
    print(torch.scatter_add(y, 0, index, x))  # [[13, 16], [3, 4]]
    print(torch.scatter_add(y, 1, index, x))  # [[12, 2], [18, 4]]
    print(torch.scatter_add(y, 0, index, 8))  # [[17, 18], [3, 4]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int
        The dimension of index values.
    index : dragon.vm.torch.Tensor
        The index tensor.
    src : Union[dragon.vm.torch.Tensor, number]
        The tensor to add from.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    """
    if not isinstance(src, Tensor):
        src = constant_ops.full_like(
            index, src, dtype=input.dtype, device=input.device)
    return Function.apply(
        'ScatterAdd', input.device, [input, index, src],
        outputs=[out], axis=dim)


def split(tensor, split_size_or_sections, dim=0, copy=True):
    """Split input into chunks along the given dimension.

    Either size of every chunk or each chunk will be accepted:

    ```python
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    # Shape: (6,) -> (4,), (2,)
    print(torch.split(x, split_size_or_sections=4))
    # Shape: (6,) -> (5,), (1,)
    print(torch.split(x, split_size_or_sections=(5, 1)))
    ```

    :attr:`dim` can be negative:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(torch.split(x, 2, dim=1))
    print(torch.split(x, 2, dim=-1))  # Equivalent
    ```

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    split_size_or_sections : Union[int, Sequence[int]
        The number or size of chunks.
    dim : int, optional, default=0
        The dimension to split.
    copy : bool, optional, default=True
        Copy or create the views of input.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The output tensors.

    """
    if nest.is_sequence(split_size_or_sections):
        size_splits = split_size_or_sections
        num_splits = len(split_size_or_sections)
    else:
        size = tensor.shape[dim]
        if size % split_size_or_sections == 0:
            num_splits = size // split_size_or_sections
            size_splits = [split_size_or_sections] * num_splits
        else:
            num_splits = size // split_size_or_sections + 1
            size_splits = [split_size_or_sections] * num_splits
            size_splits[-1] = size - (split_size_or_sections * (num_splits - 1))
    return Function.apply(
        'Split', tensor.device, [tensor], outputs=[None] * num_splits,
        axis=dim, num_splits=num_splits, split=size_splits, copy=copy)


def squeeze(input, dim=None, out=None):
    """Remove the dimensions of input with size 1.

    The argument ``dim`` could be negative or **None**:

    ```python
    x = torch.ones(1, 2, 2, 1)

    # Remove all matched dimensions if ``axis`` is None
    # Otherwise, only the specified axes will be removed
    print(torch.squeeze(x).shape)         # (1, 2, 2, 1) -> (2, 2)
    print(torch.squeeze(x, dim=0).shape)  # (1, 2, 2, 1) -> (2, 2, 1)

    # A negative dimension is the last-k dimension
    print(torch.squeeze(x, dim=3).shape)   # (1, 2, 2, 1) -> (1, 2, 2)
    print(torch.squeeze(x, dim=-1).shape)  # Equivalent

    # Also, ``axis`` could be a sequence of integers
    print(torch.squeeze(x, dim=[0, 3]).shape)  # (1, 2, 2, 1) -> (2, 2)
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to remove.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The new tensor.

    """
    return Function.apply(
        'Squeeze', input.device, [input], outputs=[out],
        axes=None if dim is None else nest.flatten(dim))


def stack(tensors, dim=0, out=None):
    """Stack the inputs along the given dimension.

    All the dimensions of inputs should be same:

    ```python
    x1 = torch.ones(2, 3)
    x2 = torch.zeros(2, 4)
    y = torch.stack([x1, x1])  # Ok
    z = torch.stack([x1, x2])  # Wrong
    ```

    :attr:`dim` can be negative:

    ```python
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.stack([x, x], dim=1)
    z = torch.stack([x, x], dim=-1)  # Equivalent
    ```

    Parameters
    ----------
    tensors : Sequence[dragon.vm.torch.Tensor]
        The input tensors.
    dim : int, optional, default=0
        The dimension to stack.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Stack', tensors[0].device, tensors, outputs=[out], axis=dim)


def transpose(input, dim0, dim1, out=None):
    """Return a new tensor with two dimensions swapped.

    Examples:

    ```python
    x = torch.ones(2, 3, 4)
    print(torch.transpose(x, 0, 2).shape)  # (4, 3, 2)
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim0 : int
        The first dimension to be transposed.
    dim1 : int
        The second dimension to be transposed.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dims = list(range(input.ndimension()))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return Function.apply(
        'Transpose', input.device, [input], outputs=[out],
        ndim=len(dims), perm=dims)


def tril(input, diagonal=0, out=None):
    r"""Return the lower triangular part of input.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                0, & \text{ if } j > i + k \\
                \text{input}_{ij}, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = torch.ones(3, 3)
    print(torch.tril(x))
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    diagonal : int, optional, default=0
        Diagonal above which to zero elements.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Trilu', input.device, [input], outputs=[out],
        k=diagonal, upper=False)


def triu(input, diagonal=0, out=None):
    r"""Return the upper triangular part of input.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                0, & \text{ if } j < i + k \\
                \text{input}_{ij}, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = torch.ones(3, 3)
    print(torch.triu(x))
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    diagonal : int, optional, default=0
        Diagonal below which to zero elements.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Trilu', input.device, [input], outputs=[out],
        k=diagonal, upper=True)


def unbind(input, dim=0, copy=True):
    """Unpack input into chunks along the given dimension.

    The number of outputs is equal to the size of dimension:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # Shape: (2, 3) -> (3,), (3,)
    print(torch.unbind(x, dim=0))
    ```

    :attr:`dim` can be negative:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # Shape: (2, 3) -> (2,), (2,), (2,)
    print(torch.unbind(x, dim=-1))
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional, default=0
        The dimension to unpack.
    copy : bool, optional, default=True
        Copy or create the views of input.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The output tensors.

    """
    num_outputs = input.size(dim)
    return Function.apply(
        'Split', input.device, [input], outputs=[None] * num_outputs,
        axis=dim, copy=copy, keepdims=False)


def unique(input, return_inverse=False, return_counts=False, **kwargs):
    """Return the unique elements of input.

    If ``return_inverse``, return the extra index where input mapping to:

    ```python
    x = torch.tensor([1, 2, 3, 2])
    y, index = torch.unique(x, return_inverse=True)
    print(y)  # [1, 2, 3]
    print(index)  # [0, 1, 2, 1]
    ```

    If ``return_counts``, return the extra counts of output:

    ```python
    x = torch.tensor([1, 2, 3, 2])
    y, counts = torch.unique(x, return_counts=True)
    print(y)  # [1, 2, 3]
    print(counts)  # [1, 2, 1]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    return_inverse : bool, optional, default=False
        Return the inverse index or not.
    return_counts : bool, optional, default=False
        Return the counts or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.
    dragon.vm.torch.Tensor, optional
        The inverse index tensor.
    dragon.vm.torch.Tensor, optional
        The counts tensor.

    """
    if 'sorted' in kwargs:
        kwargs.pop('sorted')
    num_outputs = 1
    if return_inverse:
        num_outputs += 1
    if return_counts:
        num_outputs += 1
    return Function.apply(
        'Unique', input.device, [input], outputs=[None] * num_outputs,
        return_inverse=return_inverse, return_counts=return_counts)


def unsqueeze(input, dim, out=None):
    """Expand the dimensions of input with size 1.

    :attr:`dim` could be negative:

    ```python
    x = torch.ones(2, 3, 4, 5)

    # dimension is the size-1 position in output
    print(torch.unsqueeze(x, dim=0).shape)  # (2, 3, 4, 5) -> (1, 2, 3, 4, 5)
    print(torch.unsqueeze(x, dim=1).shape)  # (2, 3, 4, 5) -> (2, 1, 3, 4, 5)

    # A negative dimension is the last-k dimension
    print(torch.unsqueeze(x, dim=4).shape)   # (2, 3, 4, 5) -> (2, 3, 4, 5, 1)
    print(torch.unsqueeze(x, dim=-1).shape)  # Equivalent

    # Also, dimension could be a sequence of integers
    print(torch.unsqueeze(x, dim=(-1, -3)).shape)  # (2, 3, 4, 5) -> (2, 3, 4, 1, 5, 1)
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]]
        The position to insert the new dimension.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The new tensor.

    """
    return Function.apply(
        'Unsqueeze', input.device, [input], outputs=[out],
        axes=None if dim is None else nest.flatten(dim))


def where(condition, x, y):
    r"""Select the elements from two branches under the condition.

    .. math::
        \text{out}_{i} =
            \begin{cases}
                \text{x}_{i}, & \text{ if } \text{condition}_{i} \\
                \text{y}_{i}, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    condition : dragon.vm.torch.Tensor
        The condition tensor.
    x : dragon.vm.torch.Tensor
        The elements for ``True`` branch.
    y : dragon.vm.torch.Tensor
        The elements for ``False`` branch.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Where', condition.device, [condition, x, y])
