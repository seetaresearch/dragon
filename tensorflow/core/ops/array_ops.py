# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_ops.py>
#
# ------------------------------------------------------------
"""Array ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.autograph.tensor import TensorRef
from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.ops import array_ops
from dragon.core.ops import control_flow_ops
from dragon.core.ops import init_ops
from dragon.core.ops import vision_ops


def broadcast_to(input, shape, name=None):
    """Broadcast input according to a given shape.

    The length of ``shape`` could either be less or
    more than the number of input dimensions:

    ```python
    a = tf.constant([[1], [2], [3]])
    # Shape: (3, 1) -> (3, 2)
    print(tf.broadcast_to(a, shape=(3, 2)))
    print(tf.broadcast_to(a, shape=(2,)))     # Equivalent
    print(tf.broadcast_to(a, shape=(-1, 2)))  # Equivalent

    # Shape: (3,) -> (1, 3) -> (2, 3)
    b = tf.constant([1, 2, 3])
    print(tf.broadcast_to(b, shape=(2, 3)))

    # Wrong remapping shape: (3,) -> (6,)
    # Only the dimension with size 1 could broadcast
    print(tf.broadcast_to(b, shape=(6,)))
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    shape : Sequence[Union[int, dragon.Tensor]]
        The output shape to broadcast to.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return array_ops.broadcast_to(input, shape, name=name)


def concat(values, axis, name='concat'):
    r"""Concatenate the values along the given axis.

    All dimensions except the ``axis`` should be same:

    ```python
    x1 = tf.ones(shape=(2, 3))
    x2 = tf.zeros(shape=(2, 4))
    y = tf.concat([x1, x2], axis=1)  # Ok
    z = tf.concat([x1, x2], axis=0)  # Wrong
    ```

    The ``axis`` can be negative representing the last-k axis:

    ```python
    y = tf.concat([x1, x2], axis=1)
    z = tf.concat([x1, x2], axis=-1)  # z == y
    ```

    Parameters
    ----------
    values : Sequence[dragon.Tensor]
        The input tensors.
    axis : int
        The axis to concatenate
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return array_ops.concat(values, axis=axis, name=name)


def depth_to_space(input, block_size, data_format='NHWC', name=None):
    """Rearrange depth data into spatial blocks.

    Examples:

    ```python
    n, h, w, c, bs = 1, 1, 1, 4, 2
    x = tf.range(n * h * w * c).reshape((n, h, w, c))
    y = tf.reshape(x, (n, h, w, bs, bs, c // (bs ** 2)))
    y = tf.transpose(y, (0, 1, 3, 2, 4, 5))
    y = tf.reshape(y, (n, h * bs, w * bs, c // (bs ** 2)))
    z = tf.nn.depth_to_space(x, 2)  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    block_size : int, required
        The size of spatial block.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return vision_ops.depth_to_space(
        input,
        block_size=block_size,
        data_format=data_format,
        name=name,
    )


def expand_dims(input, axis, name=None):
    """Expand the dimensions of input with size 1.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.ones((2, 3, 4, 5))

    # ``axis`` determines the size-1 position in output
    print(tf.expand_dims(x, axis=0).shape)  # (2, 3, 4, 5) -> (1, 2, 3, 4, 5)
    print(tf.expand_dims(x, axis=1).shape)  # (2, 3, 4, 5) -> (2, 1, 3, 4, 5)

    # A negative axis is the last-k axis
    print(tf.expand_dims(x, axis=4).shape)   # (2, 3, 4, 5) -> (2, 3, 4, 5, 1)
    print(tf.expand_dims(x, axis=-1).shape)  # Equivalent

    # Also, ``axis`` could be a sequence of integers
    print(tf.expand_dims(x, axis=[-1, -3]).shape)  # (2, 3, 4, 5) -> (2, 3, 4, 1, 5, 1)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]]
        The axis to insert the new dimension(s).
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return array_ops.expand_dims(input, axis=axis, name=name)


def fill(dims, value=0, dtype=None, name=None):
    r"""Return a tensor filled with the scalar value.

    .. math:: \text{out} \leftarrow \text{value}

    Examples:

    ```python
    x = tf.fill([2, 3], value=9)
    ```

    Parameters
    ----------
    dims : Sequence[Union[int, dragon.Tensor]]
        The shape of the tensor.
    value : number, optional, default=0
        The value to fill.
    dtype : str, optional
        The optional data type.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if dtype is None:
        dtype = numpy.array(value).dtype
        if dtype == numpy.int64:
            dtype = 'int32'
        elif dtype == numpy.float64:
            dtype = 'float32'
    return init_ops.fill(shape=dims, value=value, dtype=dtype, name=name)


def gather(params, indices, axis=0, name=None):
    """Select the elements according to the index along the given axis.

    ``indices`` could be a **int64** tensor or a sequence with integers:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(tf.gather(x, [0, 1]))
    print(tf.gather(x, tf.constant([0, 1], tf.int64)))
    ```

    More than one axis could be specified for ``indices``:

    ```python
    # The number of ``axis`` should less than ``rank(indices)``
    # And these axes should be continuous
    print(tf.gather(x, [0, 1], axis=[0, 1]))
    ```

    Parameters
    ----------
    params : dragon.Tensor
        The tensor to provide elements.
    indices : Union[Sequence[int], dragon.Tensor]
        The indices of select.
    axis : Union[int, Sequence[int]], optional, default=0
        The axis where the indices aligned.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return array_ops.index_select(params, indices, axis=axis, name=name)


def identity(input, name=None):
    """Return a new tensor copying the content of input.

    Examples:

    ```python
    # Copy ``x`` to ``xx``
    x = tf.zeros(shape=(2, 3))
    xx = tf.identity(x)

    # ``x`` != ``xx``
    x += 1
    print(x)
    print(xx)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return control_flow_ops.copy(input, name=name if name else 'Identity')


def ones(shape, dtype='float32', name=None):
    r"""Return a tensor filled with ones.

    .. math:: \text{out} \leftarrow 1

    ```python
    x = tf.ones(shape=(2, 3), dtype=tf.float32)
    ```

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The shape of the tensor.
    dtype : str, optional, default='float32'
        The optional data type.
    name : str, optional
        The operation name.

    """
    return init_ops.fill(shape, value=1, dtype=dtype, name=name)


def ones_like(input, dtype='float32', name=None):
    r"""Return a tensor of ones with shape as the other.

    .. math:: \text{out} \leftarrow 1

    Examples:

    ```python
    x = tf.ones(shape=(2, 3))
    y = tf.ones_like(x)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The tensor to hint the shape.
    dtype : str, optional, default='float32'
        The optional data type.
    name : str, optional
        The operation name.

    """
    return init_ops.ones_like(input, dtype=dtype, name=name)


def one_hot(
    indices,
    depth,
    on_value=1,
    off_value=0,
    name=None,
):
    r"""Return the one-hot representation for input.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                \text{off\_value}, & \text{ if } \text{input}_{i} \neq j \\
                \text{on\_value}, & \text{ otherwise }
            \end{cases}

    The max value of input, i.e., the ``depth`` should be specified:

    ```python
    x = tf.constant([0, 1, 2, 3], tf.int64)
    print(tf.one_hot(x, depth=5))  # depth >= 4 will be ok
    ```

    You can also set the ``on_value`` or ``off_value``:

    ```python
    print(tf.one_hot(x, depth=4, on_value=2, off_value=3))
    ```

    Parameters
    ----------
    indices : dragon.Tensor
        The input tensor.
    depth : int
        The depth of representation.
    on_value : int, optional, default=1
        The value for equal branch.
    off_value : int, optional, default=0
        The value for not-equal branch.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return array_ops.one_hot(
        indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        name=name,
    )


def pad(
    tensor,
    paddings,
    mode='CONSTANT',
    constant_values=0,
    name=None,
):
    r"""Pad the input according to the given sizes.

    The ``paddings`` should be a sequence with :math:`N` tuples,
    where :math:`N` is the number of input dimensions:

    ```python
    x = tf.ones(shape=(2, 3))
    print(tf.pad(x, [[0, 1], [1, 0]]))  # Ok, (2, 3) -> (3, 4)
    print(tf.pad(x, [[0, 1]]))  # Wrong
    ```

    Following padding modes are supported:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]])

    # ConstantPad
    print(tf.pad(x, [[0, 1], [1, 0]], 'CONSTANT', 9))

    # ReflectPad
    print(tf.pad(x, [[0, 1], [1, 0]], 'REFLECT'))

    # SymmetricPad (EdgePad)
    print(tf.pad(x, [[0, 1], [1, 0]], 'SYMMETRIC'))
    ```

    Parameters
    ----------
    tensor : dragon.Tensor
        The input tensor.
    paddings : Sequence[Tuple[int]]
        The begins and ends of padding.
    mode : {'CONSTANT', 'REFLECT', 'SYMMETRIC'}, optional
        The padding mode.
    constant_values : int, optional, default=0
        The constant value in ``CONSTANT`` mode.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return array_ops.pad(
        tensor,
        pads=paddings,
        mode={
            'CONSTANT': 'CONSTANT',
            'REFLECT': 'REFLECT',
            'SYMMETRIC': 'EDGE',
        }[mode.upper()],
        value=constant_values,
        name=name,
    )


def placeholder(dtype=None, shape=None, name=None):
    """Return a symbolic tensor as the placeholder.

    Parameters
    ----------
    dtype : str, optional
        The data type provided to cast the input.
    shape : Sequence[int], optional
        The optional tensor shape.
    name : str, optional
        The optional tensor name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    # Construct a tensor from the explicit name
    return TensorRef(
        workspace.get_workspace().unique_name(
            context.get_name_scope() + name if name else 'Placeholder',
            suffix=':0', namespace='Tensor'),
        dtype=dtype if dtype else dtype,
        shape=shape,
    ).constant()


def reshape(tensor, shape, name=None):
    """Change the dimensions of input.

    Examples:

    ```python
    # Provide a determined value for each dimension if possible
    x = tf.ones(shape=(1, 2, 3, 4))
    print(tf.reshape(x, shape=[6, 4]).shape)  # [6, 4]

    # Set the existing dimensions to ``0`` if it unchanged
    print(tf.reshape(x, shape=[0, 0, 12]).shape)  # [1, 2, 12]
    print(tf.reshape(x, shape=[0, 0, 0, 0]).shape)  # [1, 2, 3, 4]
    print(tf.reshape(x, shape=[0, 0, 0, 0, 0]).shape)  # Wrong

    # You can also set ``-1`` once to infer the value
    print(tf.reshape(x, shape=[-1, 4]).shape)  # [6, 4]
    print(tf.reshape(x, shape=[-1, -1]).shape)  # Wrong
    ```

    Parameters
    ----------
    tensor : dragon.Tensor
        The input tensor.
    shape : Union[Sequence[int], dragon.Tensor]
        The output shape.
    name : str, optional
        The operation name.

    """
    return array_ops.reshape(tensor, shape=shape, name=name)


def shape(input, name=None):
    """Return the shape of input.

    Examples:

    ```python
    x = tf.ones((2, 3))
    print(x.shape)  # Return a sequence
    print(tf.shape(x))  # Return a tensor
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The tensor shape.

    """
    return array_ops.shape(input, name=name)


def slice(input_, begin, size, name=None):
    """Select the elements according to the given sections.

    The section is hinted by ``[begin[i], begin[i] + size[i])``:

    ```python
    x = tf.constant([[[0, 1, 2], [3, 4, 5]]])
    print(tf.slice(x, [0, 1, 2], [1, 1, 1]))  # [[[5]]]
    print(x[0:1, 1:2:, 2:3])  # Equivalent
    ```

    The ``size`` accepts value **-1** or **0**:

    ```python
    x = tf.constant([[[0, 1, 2], [3, 4, 5]]])

    # Set ``0`` to squeeze dimensions with size 1
    print(tf.slice(x, [0, 1, 2], [0, 0, 0]))  # 5

    # Set ``-1`` to take all the remained elements
    print(tf.slice(x, [0, 0, 0], [-1, -1, -1]))  # [[[0, 1, 2], [3, 4, 5]]]
    ```

    Parameters
    ----------
    input_ : dragon.Tensor
        The input tensor.
    begin : Union[Sequence[int], dragon.Tensor]
        The start location for each dimension.
    size : Union[Sequence[int], dragon.Tensor]
        The number of elements sliced from start.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return array_ops.slice(input_, starts=begin, sizes=size, name=name)


def space_to_depth(input, block_size, data_format='NHWC', name=None):
    """Rearrange blocks of spatial data into depth.

    Examples:

    ```python
    n, h, w, c, bs = 1, 2, 2, 2, 2
    x = tf.range(n * h * w * c).reshape((n, h, w, c))
    y = tf.reshape(x, (n, h // bs, bs, w // bs, bs, c))
    y = tf.transpose(y, (0, 1, 3, 2, 4, 5))
    y = tf.reshape(y, (n, h // bs, w // bs, bs * bs * c))
    z = tf.nn.space_to_depth(x, 2)  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    block_size : int, required
        The size of spatial block.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return vision_ops.space_to_depth(
        input,
        block_size=block_size,
        data_format=data_format,
        name=name,
    )


def split(
    value,
    num_or_size_splits,
    axis=0,
    name=None,
):
    """Split input into chunks along the given axis.

    Either number or size of splits will be accepted:

    ```python
    x = tf.constant([[1, 2], [3, 4], [5, 6]])
    # Shape: (3, 2) -> (2, 2), (1, 2)
    print(tf.split(x, num_or_size_splits=2))
    # Shape: (3, 2) -> (1, 2), (2, 2)
    print(tf.split(x, num_or_size_splits=(1, 2)))
    ```

    The ``axis`` can be negative representing the last-k axis:

    ```python
    x = tf.constant([[1, 2], [3, 4], [5, 6]])
    print(tf.split(x, 2, axis=1))
    print(tf.split(x, 2, axis=-1))  # Equivalent
    ```

    Parameters
    ----------
    value : dragon.Tensor
        The input tensor.
    num_or_size_splits: Union[int, Sequence[int]]
        The number or size of chunks.
    axis : int, optional, default=0
        The axis to split.
    name : str, optional
        The operation name.

    Returns
    -------
    Sequence[dragon.Tensor]
        The outputs.

    """
    return array_ops.split(value, num_or_size_splits, axis, name=name)


def squeeze(input, axis=None, name=None):
    """Remove the dimensions of input with size 1.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.ones((1, 2, 2, 1))

    # Remove all matched dimensions if ``axis`` is None
    # Otherwise, only the specified axes will be removed
    print(tf.squeeze(x).shape)  # (1, 2, 2, 1) -> (2, 2)
    print(tf.squeeze(x, axis=0).shape)  # (1, 2, 2, 1) -> (2, 2, 1)

    # A negative axis is the last-k axis
    print(tf.squeeze(x, axis=3).shape)  # (1, 2, 2, 1) -> (1, 2, 2)
    print(tf.squeeze(x, axis=-1).shape)  # Equivalent

    # Also, ``axis`` could be a sequence of integers
    print(tf.squeeze(x, axis=[0, 3]).shape)  # (1, 2, 2, 1) -> (2, 2)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to remove.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return array_ops.squeeze(input, axis=axis, name=name)


def tile(input, multiples, name=None):
    """
    Create a tile from a list of arrays.

    Args:
        input: (array): write your description
        multiples: (bool): write your description
        name: (str): write your description
    """
    return array_ops.tile(input, repeats=multiples, name=name)


def transpose(a, perm=None, name=None):
    r"""Permute the dimensions of input.

    Examples:

    ```python
    # Provide the permutation for all axes
    x = tf.ones(shape=(2, 3, 4))
    print(tf.transpose(x, (0, 2, 1)).shape)  # (2, 4, 3)

    # Or dimensions will be simply inverse
    print(tf.transpose(x).shape)  # (4, 3, 2)
    ```

    Parameters
    ----------
    a : dragon.Tensor
        The input tensor.
    perm : Sequence[Union[int, dragon.Tensor]]
        The output permutation.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return array_ops.transpose(a, perm=perm, name=name)


def unique(x, name=None, **kwargs):
    """Return the unique elements of input.

    Unique elements and index where input mapping to are returned:

    ```python
    x = tf.constant([1, 2, 3, 2])
    y, index = tf.unique(x)
    print(y)  # [1, 2, 3]
    print(index)  # [0, 1, 2, 1]
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.
    dragon.Tensor
        The inverse index tensor.

    """
    if 'out_idx' in kwargs:
        kwargs.pop('out_idx')
    return array_ops.unique(x, return_inverse=True, name=name)


def unique_with_counts(x, name=None, **kwargs):
    """Return the unique elements of input with counts.

    Unique elements, remapping index and counts are returned:

    ```python
    x = tf.constant([1, 2, 3, 2])
    y, index, counts = tf.unique_with_counts(x)
    print(y)  # [1, 2, 3]
    print(index)  # [0, 1, 2, 1]
    print(counts)  # [1, 2, 1]
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.
    dragon.Tensor
        The inverse index tensor.
    dragon.Tensor
        The counts tensor.

    """
    if 'out_idx' in kwargs:
        kwargs.pop('out_idx')
    return array_ops.unique(
        x,
        return_inverse=True,
        return_counts=True,
        name=name,
    )


def zeros(shape, dtype='float32', name=None):
    r"""Return a tensor filled with zeros.

    .. math:: \text{out} \leftarrow 0

    ```python
    x = tf.zeros(shape=(2, 3), dtype=tf.float32)
    ```

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The shape of the tensor.
    dtype : str, optional, default='float32'
        The optional data type.
    name : str, optional
        The operation name.

    """
    return init_ops.fill(shape, value=0., dtype=dtype, name=name)


def zeros_like(input, dtype='float32', name=None):
    r"""Return a tensor of zeros with shape as the other.

    .. math:: \text{out} \leftarrow 0

    Examples:

    ```python
    x = tf.zeros(shape=(2, 3))
    y = tf.zeros_like(x)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The tensor to hint the shape.
    dtype : str, optional, default='float32'
        The optional data type.
    name : str, optional
        The operation name.

    """
    return init_ops.zeros_like(input, dtype=dtype, name=name)
