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
"""Array functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest
from dragon.vm.torch.core.ops import utils
from dragon.vm.torch.core.ops.array import _functions
from dragon.vm.torch.core.tensor import Tensor


def argmax(input, dim=None, keepdim=False, out=None):
    """Return the index of maximum elements along the given dimension.

    The argument ``dim`` could be negative or **None**:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative ``dim`` is the last-k axis
    print(torch.argmax(x, 1))
    print(torch.argmax(x, -1))  # Equivalent

    # If ``dim`` is None, the vector-style reduction
    # will be applied to return a scalar index
    print(torch.argmax(x))  # 5
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The index of maximum elements.

    """
    return _arg_reduce(input, 'ArgMax', dim, keepdim, out)


def argmin(input, dim=None, keepdim=False, out=None):
    """Return the index of minimum elements along the given dimension.

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative ``dim`` is the last-k axis
    print(torch.argmin(x, 1))
    print(torch.argmin(x, -1))  # Equivalent

    # If ``dim`` is None, the vector-style reduction
    # will be applied to return a scalar index
    print(torch.argmin(x))  # 0
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The index of minimum elements.

    """
    return _arg_reduce(input, 'ArgMin', dim, keepdim, out)


def argsort(input, dim=-1, descending=False):
    """Return the index of sorted elements along the given dimension.

    By default, the last dimension is chosen:

    ```python
    x = torch.tensor([[1, 2, 3], [3, 2, 1]])
    index1 = torch.argsort(x)
    index2 = torch.argsort(x, dim=1)  # Equivalent
    ```

    Sort in the descending order if ``descending`` is ``True``:

    ```python
    x = torch.tensor([1, 2, 3])
    index1 = torch.argsort(-x)
    index2 = torch.argsort(x, descending=True)  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional, default=-1
         The dimension to sort elements.
    descending : bool, optional, default=False
        Sort in the descending order or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return sort(input, dim, descending)[1]


def assign(out, starts, sizes, input):
    """
    Assigns tensor to a tensor.

    Args:
        out: (array): write your description
        starts: (todo): write your description
        sizes: (int): write your description
        input: (array): write your description
    """
    if not isinstance(input, Tensor):
        input = utils.scalar_to_tensor(
            input,
            dtype=out.dtype,
            device=out.device,
        )
    return _functions.Assign \
        .instantiate(out.device, ndim=len(starts)) \
        .apply(out, starts, sizes, input)


def cast(input, dtype='float32', inplace=False):
    """Cast the data type of input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input.
    dtype : str, optional, default='float32'
        The data type to cast to.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Cast \
        .instantiate(input.device, dtype=dtype) \
        .apply(input, inplace)


def cat(seq, dim=0, out=None):
    """Concatenate the inputs along the given dimension.

    Parameters
    ----------
    seq : Sequence[dragon.vm.torch.Tensor]
        The input sequence.
    dim : int, optional
        The dim to concatenate.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Concat \
        .instantiate(seq[0].device, axis=dim) \
        .apply(seq, out)


def channel_affine(input, weight, bias=None, dim=0, out=None):
    """Apply affine transformation along the channels.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The optional bias.
    dim : int, optional, default=0
        The start dimension to transform.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.ChannelAffine \
        .instantiate(
            input.device,
            axis=dim,
            num_axes=weight.ndimension(),
        ).apply(input, weight, bias, out)


def channel_normalize(
    input,
    mean,
    std,
    dim=-1,
    dtype='float32',
    dims=None,
):
    """Normalize channels with mean and standard deviation.

    The ``dim`` can be negative representing the last-k dimension:

    ```python
    m = s = (1., 1., 1.)
    x = torch.tensor([1, 2, 3])
    print(torch.channel_normalize(x, m, s, dim=0))   # [0., 1., 2.]
    print(torch.channel_normalize(x, m, s, dim=-1))  # Equivalent
    ```

    If ``dims`` is provided, ``dim`` is selected from the output layout:

    ```python
    m, s = (1., 2., 3.), (1., 1., 1.)
    x = torch.tensor([[1, 2, 3]])
    # Provided 3 values to normalize the last dimension
    # with length 1, only the first value will be taken
    print(torch.channel_normalize(x, m, s, dims=(1, 0)))  # [[0.], [1.], [2.]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    mean : Sequence[float], required
        The mean to subtract.
    std : Sequence[float], required
        The standard deviation to divide.
    dim : int, optional, default=-1
        The dimension to normalize.
    dtype : str, optional, default='float32'
        The output data type.
    dims : Sequence[int], optional
        The order of output dimensions.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.ChannelNormalize \
        .instantiate(
            input.device,
            axis=dim,
            ndim=len(dims) if dims is not None else 0,
            mean=mean,
            std=std,
            dtype=dtype,
        ).apply(input, dims)


def channel_shuffle(input, dim=0, groups=1, out=None):
    """Shuffle channels between a given number of groups.
    `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional, default=0
        The dimension of channels.
    groups : int, optional, default=1
        The number of groups.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.ChannelShuffle \
        .instantiate(
            input.device,
            axis=dim,
            group=groups,
        ).apply(input, out)


def chunk(tensor, chunks, dim=0):
    """Split input into a specific number of chunks.

    Examples:

    ```python
    x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # Shape: (3, 2) -> (2, 2), (1, 2)
    print(torch.chunk(x, chunks=2))
    ```

    The ``dim`` can be negative representing the last-k axis:

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

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The output tensors.

    """
    return _functions.Split \
        .instantiate(
            tensor.device,
            axis=dim,
            size_splits=None,
        ).apply(tensor, chunks)


def cumsum(input, dim, out=None):
    """Compute the cumulative sum of elements along the given dimension.

    The argument ``dim`` could be negative:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative dimension is the last-k dimension
    print(torch.cumsum(x, 1))  # [[1, 3, 6], [4, 9, 15]]
    print(torch.cumsum(x, -1))  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int
        The cumulative dimension.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Cumulative \
        .instantiate(
            input.device,
            operation='Sum',
            axis=dim,
        ).apply(input, out)


def expand(input, sizes):
    """Broadcast input according to given sizes.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    sizes : Sequence[int]
        The output dimensions to broadcast to.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    sizes = nest.flatten(sizes)
    return _functions.Expand \
        .instantiate(input.device, ndim=len(sizes)) \
        .apply(input, sizes)


def flatten(input, start_dim=0, end_dim=-1, out=None):
    """Return a tensor with dimensions flattened.

    The argument ``start_dim`` and ``end_dim`` could be negative:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(torch.flatten(x, start_dim=0, end_dim=-1))
    print(torch.flatten(x, start_dim=0, end_dim=1))  # Equivalent
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
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if end_dim == -1:
        num_axes = -1
    else:
        while end_dim < 0:
            end_dim += input.ndimension()
        while start_dim < 0:
            start_dim += input.ndimension()
        num_axes = end_dim - start_dim + 1
    return _functions.Flatten \
        .instantiate(
            input.device,
            axis=start_dim,
            num_axes=num_axes,
        ).apply(input, out)


def index_select(input, dim, index, out=None):
    """Select the elements along the given dim using index.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]]
        The dim(s) to select.
    index : dragon.vm.torch.Tensor
        The index tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dim = nest.flatten(dim)
    dim.sort()
    if dim[-1] != (dim[0] + len(dim) - 1):
        raise ValueError('The <dim> should be a continuous sequence.')
    return _functions.IndexSelect \
        .instantiate(
            utils.unify_devices([input, index]),
            axis=dim[0],
            num_axes=len(dim),
        ).apply(input, index, out)


def masked_fill(out, mask, value):
    """Fill tensor with the given value where ``mask`` is **1**.

    Parameters
    ----------
    out : dragon.vm.torch.Tensor
        The tensor to fill value.
    mask : dragon.vm.torch.Tensor
        The boolean mask.
    value : number
        The value to fill.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if not isinstance(value, Tensor):
        value = utils.scalar_to_tensor(
            value,
            dtype=out.dtype,
            device=out.device,
        )
    return _functions.MaskedAssign \
        .instantiate(out.device).apply(out, mask, value)


def masked_select(input, mask, out=None):
    """Select the input elements where mask is **1**.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    mask : dragon.vm.torch.Tensor
        The mask for selecting.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.MaskedSelect \
        .instantiate(
            utils.unify_devices([input, mask]),
        ).apply(input, mask, out)


def max(input, dim=None, keepdim=False, out=None):
    """Compute the max value of elements along the given dimension.

    The argument ``dim`` could be negative or **None**:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative ``dim`` is the last-k axis
    print(torch.max(x, 1))
    print(torch.max(x, -1))  # Equivalent

    # If ``dim`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(torch.max(x))  # Result is 3

    # Also, ``dim`` could be a sequence of integers
    print(torch.max(x, [0, 1]))  # Result is 3
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimensions or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _reduce(input, 'Max', dim, keepdim, out)


def mean(input, dim=None, keepdim=False, out=None):
    """Compute the mean value of elements along the given dimension.

    The argument ``dim`` could be negative or **None**:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative ``dim`` is the last-k axis
    print(torch.mean(x, 1))
    print(torch.mean(x, -1))  # Equivalent

    # If ``dim`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(torch.mean(x))  # Result is 3

    # Also, ``dim`` could be a sequence of integers
    print(torch.mean(x, [0, 1]))  # Result is 3
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimensions or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _reduce(input, 'Mean', dim, keepdim, out)


def min(input, dim=None, keepdim=False, out=None):
    """Compute the min value of elements along the given dimension.

    The argument ``dim`` could be negative or **None**:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative ``dim`` is the last-k axis
    print(torch.min(x, 1))
    print(torch.min(x, -1))  # Equivalent

    # If ``dim`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(torch.min(x))  # Result is 1

    # Also, ``dim`` could be a sequence of integers
    print(torch.min(x, [0, 1]))  # Result is 1
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimensions or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _reduce(input, 'Min', dim, keepdim, out)


def multinomial(input, num_samples, epsilon=0, out=None):
    """Return a tensor with index sampled from multinomial distribution.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    num_samples : int
        The number of samples in each row.
    epsilon : float, optional, default=0
        The epsilon value to apply epsilon-greedy strategy.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Multinomial \
        .instantiate(
            input.device,
            num_samples=num_samples,
            epsilon=float(epsilon),
        ).apply(input, out)


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
    return slice(input, starts, sizes)


def nonzero(input, out=None):
    r"""Return the index of non-zero elements.

    .. math:: \text{out} = \{i\}, \text{ if } \text{input}_{i} \neq 0

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.NonZero.instantiate(input.device).apply(input, out)


def one_hot(input, depth):
    r"""Return the one-hot representation for input.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                0, & \text{ if } \text{input}_{i} \neq j \\
                1, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    depth : int
        The depth of channels.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.OneHot.instantiate(input.device, depth=depth).apply(input)


def permute(input, dims):
    """Return a new tensor with the specific order of dimensions.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dims : Sequence[int]
        The new order of dimensions.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dims = nest.flatten(dims)
    return _functions.Transpose \
        .instantiate(input.device, ndim=len(dims)) \
        .apply(input, dims)


def repeat(input, sizes):
    """Repeat elements along the specified dimensions.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    sizes : Sequence[int]
        The number of times to repeat.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    sizes = nest.flatten(sizes)
    return _functions.Tile \
        .instantiate(input.device, ndim=len(sizes)) \
        .apply(input, sizes)


def reshape(input, shape, out=None):
    """Change the shape of input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    shape : Sequence[int]
        The new shape.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The new tensor.

    """
    shape = nest.flatten(shape)
    return _functions.Reshape \
        .instantiate(input.device, ndim=len(shape)) \
        .apply(input, shape, out)


def slice(input, starts, sizes):
    """
    Slice a slice of a slice.

    Args:
        input: (array): write your description
        starts: (int): write your description
        sizes: (int): write your description
    """
    return _functions.Slice \
        .instantiate(input.device, ndim=len(starts)) \
        .apply(input, starts, sizes)


def sort(input, dim=-1, descending=False, out=None):
    """Return the sorted elements along the given dimension.

    By default, the last dimension is chosen:

    ```python
    x = torch.tensor([[1, 2, 3], [3, 2, 1]])
    value1, index1 = torch.sort(x)
    value2, index2 = torch.sort(x, dim=1)  # Equivalent
    ```

    Sort in the descending order if ``descending`` is ``True``:

    ```python
    x = torch.tensor([1, 2, 3])
    _, index1 = torch.sort(-x)
    _, index2 = torch.sort(x, descending=True)  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int, optional, default=-1
         The dimension to sort elements.
    descending : bool, optional, default=False
        Sort in the descending order or not.
    out : Sequence[dragon.vm.torch.Tensor], optional
        The optional output value and index.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The value and index tensor.

    """
    return _functions.Sort \
        .instantiate(
            input.device,
            axis=dim,
            descending=descending,
        ).apply(input, out if out else (None, None))


def split(tensor, split_size_or_sections, dim=0):
    """Split input into chunks along the given dimension.

    Either size of every chunk or each chunk will be accepted:

    ```python
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    # Shape: (6,) -> (4,), (2,)
    print(torch.split(x, split_size_or_sections=4))
    # Shape: (6,) -> (5,), (1,)
    print(torch.split(x, split_size_or_sections=(5, 1)))
    ```

    The ``dim`` can be negative representing the last-k axis:

    ```python
    x = torch.tensor([[1, 2], [3, 4], [5, 6]])
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

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The output tensors.

    """
    if nest.is_sequence(split_size_or_sections):
        num_splits = len(split_size_or_sections)
        size_splits = split_size_or_sections
    else:
        size = tensor.shape[dim]
        if size % split_size_or_sections == 0:
            num_splits = size // split_size_or_sections
            size_splits = [split_size_or_sections] * num_splits
        else:
            num_splits = size // split_size_or_sections + 1
            size_splits = [split_size_or_sections] * num_splits
            size_splits[-1] = size - (split_size_or_sections * (num_splits - 1))
    return _functions.Split \
        .instantiate(
            tensor.device,
            axis=dim,
            size_splits=size_splits,
        ).apply(tensor, num_splits)


def squeeze(input, dim=None, out=None):
    """Remove the dimensions of input with size 1.

    The argument ``dim`` could be negative or **None**:

    ```python
    x = torch.ones(1, 2, 2, 1)

    # Remove all matched dimensions if ``axis`` is None
    # Otherwise, only the specified axes will be removed
    print(torch.squeeze(x).shape)         # (1, 2, 2, 1) -> (2, 2)
    print(torch.squeeze(x, dim=0).shape)  # (1, 2, 2, 1) -> (2, 2, 1)

    # A negative axis is the last-k axis
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
    axes = None if dim is None else nest.flatten(dim)
    return _functions.Squeeze \
        .instantiate(
            input.device,
            axes=axes,
        ).apply(input, out)


def stack(seq, dim=0, out=None):
    """Stack the inputs along the given dimension.

    Parameters
    ----------
    seq : sequence of dragon.vm.torch.Tensor
        The sequence.
    dim : int, optional, default=0
        The dim to stack.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Stack \
        .instantiate(
            utils.unify_devices(seq),
            axis=dim,
        ).apply(seq, out)


def sum(input, dim=None, keepdim=False, out=None):
    """Compute the sum value of elements along the given dimension.

    The argument ``dim`` could be negative or **None**:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative ``dim`` is the last-k axis
    print(torch.sum(x, 1))
    print(torch.sum(x, -1))  # Equivalent

    # If ``dim`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(torch.sum(x))  # Result is 21

    # Also, ``dim`` could be a sequence of int
    print(torch.sum(x, [0, 1]))  # Result is 21
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimensions or not.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _reduce(input, 'Sum', dim, keepdim, out)


def topk(input, k, dim=-1, largest=True, sorted=True, out=None):
    """Return the top-K largest or smallest elements along the given dimension.

    By default, the last dimension is chosen:

    ```python
    x = torch.tensor([[1, 2, 3], [3, 2, 1]])
    value1, index1 = torch.topk(x, k=2)
    value2, index2 = torch.topk(x, k=2, dim=1)  # Equivalent
    ```

    If ``largest`` is ``False``, the k smallest elements are returned:

    ```python
    x = torch.tensor([1, 2, 3])
    _, index1 = torch.topk(-x, 1)
    _, index2 = torch.topk(x, 1, largest=False)  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    k : int
        The number of top elements to select.
    dim : int, optional, default=-1
         The dimension to select elements.
    largest : bool, optional
        Return largest or smallest elements.
    sorted : bool, optional
        Whether to return in the sorted order.
    out : Sequence[dragon.vm.torch.Tensor], optional
        The optional output value and index.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The value and index tensor.

    """
    return _functions.TopK \
        .instantiate(
            input.device,
            k=k,
            axis=dim,
            largest=largest,
            sorted=sorted,
        ).apply(input, out if out else (None, None))


def transpose(input, dim0, dim1):
    """Return a new tensor with two dimensions swapped.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim0 : int
        The first dimension to be transposed.
    dim1 : int
        The second dimension to be transposed.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    dims = list(range(input.ndimension()))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return _functions.Transpose \
        .instantiate(input.device, ndim=len(dims)) \
        .apply(input, dims)


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
    return _functions.Unique \
        .instantiate(
            input.device,
            return_inverse=return_inverse,
            return_counts=return_counts,
        ).apply(input)


def unsqueeze(input, dim, out=None):
    """Expand the dimensions of input with size 1.

    The argument ``dim`` could be negative or **None**:

    ```python
    x = torch.ones(2, 3, 4, 5)

    # ``dim`` determines the size-1 position in output
    print(torch.unsqueeze(x, dim=0).shape)  # (2, 3, 4, 5) -> (1, 2, 3, 4, 5)
    print(torch.unsqueeze(x, dim=1).shape)  # (2, 3, 4, 5) -> (2, 1, 3, 4, 5)

    # A negative ``dim`` is the last-k axis
    print(torch.unsqueeze(x, dim=4).shape)   # (2, 3, 4, 5) -> (2, 3, 4, 5, 1)
    print(torch.unsqueeze(x, dim=-1).shape)  # Equivalent

    # Also, ``dim`` could be a sequence of integers
    print(torch.unsqueeze(x, dim=[-1, -3]).shape)  # (2, 3, 4, 5) -> (2, 3, 4, 1, 5, 1)
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]]
        The position to insert the new dimension(s).
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The new tensor.

    """
    axes = None if dim is None else nest.flatten(dim)
    return _functions.UnSqueeze \
        .instantiate(
            input.device,
            axes=axes,
        ).apply(input, out)


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
        The elements for **True** branch.
    y : dragon.vm.torch.Tensor
        The elements for **False** branch.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Where \
        .instantiate(utils.unify_devices([condition, x, y])) \
        .apply(condition, x, y)


def _arg_reduce(input, op_type, dim=None, keepdim=False, out=None):
    """The generic arg reduce function."""
    if dim is None:
        keepdim = False
    return _functions.ArgReduce \
        .instantiate(
            input.device,
            op_type=op_type,
            axis=dim,
            keep_dims=keepdim,
        ).apply(input, out)


def _reduce(input, operation, dim=None, keepdim=False, out=None):
    """The generic reduce function."""
    if dim is None:
        keepdim = False
    else:
        dim = nest.flatten(dim)
    return _functions.Reduce \
        .instantiate(
            input.device,
            axes=dim,
            keep_dims=keepdim,
            operation=operation,
        ).apply(input, out)
