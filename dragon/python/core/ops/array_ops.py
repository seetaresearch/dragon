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

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema
from dragon.core.framework import types
from dragon.core.ops import constant_ops
from dragon.core.util import nest
from dragon.core.util import six


@OpSchema.num_inputs(2)
@OpSchema.convert_arg('starts')
@OpSchema.convert_arg('sizes')
def assign(inputs, starts=None, sizes=None, copy=False, **kwargs):
    r"""Assign the value to input.

    .. math:: \text{input}[\text{start}:\text{start} + \text{size}, ...] = \text{value}

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input and value tensor.
    starts : Union[Sequence[int], dragon.Tensor]], optional
        The start location for each dimension.
    sizes : Union[Sequence[int], dragon.Tensor]], optional
        The number of elements from start.
    copy : bool, optional, default=False
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        starts = args['starts'] if starts is not None else [0]
        sizes = args['sizes'] if sizes is not None else [-1]
        return OpLib.execute(
            'Assign', inputs, outputs=[None if copy else inputs[0]],
            ndim=len(starts), starts=starts, sizes=sizes)
    return OpLib.add('Assign', **args)


@OpSchema.num_inputs(2)
def boolean_mask(inputs, **kwargs):
    """Return the elements of input where mask is true.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input and mask tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('BooleanMask', inputs)
    return OpLib.add('BooleanMask', inputs, **kwargs)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg(name='shape', name_v2='dims')
def broadcast_to(inputs, shape, **kwargs):
    """Broadcast input to the given shape.

    Length of ``shape`` could either be less or more
    than the number of input dimensions:

    ```python
    a = dragon.constant([[1], [2], [3]])
    # Shape: (3, 1) -> (3, 2)
    print(dragon.broadcast_to(a, shape=(3, 2)))
    print(dragon.broadcast_to(a, shape=(2,)))  # Equivalent

    # Shape: (3,) -> (1, 3) -> (2, 3)
    b = dragon.constant([1, 2, 3])
    print(dragon.broadcast_to(b, shape=(2, 3)))

    # Wrong remapping shape: (3,) -> (6,)
    # Only the dimension with size 1 could broadcast
    print(dragon.broadcast_to(b, shape=(6,)))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    shape : Sequence[Union[int, dragon.Tensor]]
        The output shape to broadcast to.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'Expand', inputs, ndim=len(args['dims']), dims=args['dims'])
    return OpLib.add('Expand', **args)


@OpSchema.num_inputs(1)
def channel_shuffle(inputs, axis=-1, group=1, **kwargs):
    """Apply the group shuffle to each channel of input.
    `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

    Examples:

    ```python
    x = dragon.constant([1, 2, 3, 4])
    print(dragon.nn.channel_shuffle(x, group=2))  # [1, 3, 2, 4]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The inputs.
    axis : int, optional, default=-1
        The channel axis.
    group : int, optional, default=1
        The number of shuffle groups.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('ChannelShuffle', inputs, axis=axis, group=group)
    return OpLib.add('ChannelShuffle', inputs, axis=axis, group=group, **kwargs)


@OpSchema.num_inputs(1, 2147483647)
def concat(inputs, axis=0, **kwargs):
    """Concatenate the inputs along the given axis.

    All dimensions except the :attr:`axis` should be same:

    ```python
    x1 = dragon.ones(shape=(2, 3))
    x2 = dragon.zeros(shape=(2, 4))
    y = dragon.concat([x1, x2], axis=1)  # Ok
    z = dragon.concat([x1, x2], axis=0)  # Wrong
    ```

    :attr:`axis` can be negative:

    ```python
    x = dragon.constant([[1, 2], [3, 4]])
    y = dragon.concat([x, x], axis=1)
    z = dragon.concat([x, x], axis=-1)  # Equivalent
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input tensors.
    axis : int, optional, default=0
        The axis to concatenate.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Concat', inputs, axis=axis)
    return OpLib.add('Concat', inputs, axis=axis, **kwargs)


@OpSchema.num_inputs(1)
def expand_dims(inputs, axis, copy=True, **kwargs):
    """Expand the dimensions of input with size 1.

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.ones((2, 3, 4, 5))

    # axis is the size-1 position in output
    print(dragon.expand_dims(x, axis=0).shape)  # (2, 3, 4, 5) -> (1, 2, 3, 4, 5)
    print(dragon.expand_dims(x, axis=1).shape)  # (2, 3, 4, 5) -> (2, 1, 3, 4, 5)

    # A negative axis is the last-k axis
    print(dragon.expand_dims(x, axis=4).shape)   # (2, 3, 4, 5) -> (2, 3, 4, 5, 1)
    print(dragon.expand_dims(x, axis=-1).shape)  # Equivalent

    # Also, axis could be a sequence of integers
    print(dragon.expand_dims(x, axis=[-1, -3]).shape)  # (2, 3, 4, 5) -> (2, 3, 4, 1, 5, 1)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]]
        The axis to insert the new dimension.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    axes = None if axis is None else nest.flatten(axis)
    if context.executing_eagerly():
        return OpLib.execute(
            'Unsqueeze', inputs, outputs=[None] if copy else inputs, axes=axes)
    return OpLib.add('Unsqueeze', inputs, axes=axes, **kwargs)


@OpSchema.num_inputs(1)
def flatten(inputs, axis=0, end_axis=-1, copy=True, **kwargs):
    """Flatten the input along the given axes.

    Set :attr:`keep_axes` to flatten if shape is dynamic.

    Examples:

    ```python
    x = dragon.ones((1, 2, 3, 4))
    print(dragon.flatten(x, axis=1).shape)  # (1, 24)
    print(dragon.flatten(x, axis=1, end_axis=2).shape)  # (1, 6, 4)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=0
        The first axis to flatten.
    end_axis : int, optional, default=-1
        The last axis to flatten.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Flatten', inputs, outputs=[None] if copy else inputs,
            axis=axis, end_axis=end_axis)
    return OpLib.add('Flatten', inputs, axis=axis, end_axis=end_axis, **kwargs)


@OpSchema.num_inputs(2)
def gather(inputs, axis=0, end_axis=None, **kwargs):
    """Gather elements along the given axis using index.

    Index should be a ``int64`` tensor:

    ```python
    input = dragon.constant([[1, 2, 3], [4, 5, 6]])
    index = dragon.constant([1])
    print(dragon.gather([input, index]))  # [[4, 5, 6]]
    ```

    More than one axis could be specified to gather:

    ```python
    # Along the continuous axes: [axis, end_axis]
    print(dragon.gather([input, index], axis=0, end_axis=1))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input and index tensor.
    axis : int, optional, default=0
        The first axis to gather.
    end_axis : int, optional
        The last axis to gather.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Gather', inputs, axis=axis, end_axis=end_axis)
    return OpLib.add('Gather', inputs, axis=axis, end_axis=end_axis, **kwargs)


@OpSchema.num_inputs(2)
def gather_elements(inputs, axis=0, **kwargs):
    """Gather elements along the given axis of index.

    Number of dimensions of input and index should be same.
    For 3-d input, output is gathered as:

    ```python
    out[i, j, k] = input[index[i, j, k], j, k]
    out[i, j, k] = input[i, index[i, j, k], k]
    out[i, j, k] = input[i, j, index[i, j, k]]
    ```

    Examples:

    ```python
    x = dragon.constant([[1, 2], [3, 4]])
    index = dragon.constant([[0, 0], [0, 1]])
    print(dragon.gather_elements([x, index], axis=0))  # [[1, 2], [1, 4]]
    print(dragon.gather_elements([x, index], axis=1))  # [[1, 1], [3, 4]]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input and index tensor.
    axis : int, optional, default=0
        The axis of index values.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('GatherElements', inputs, axis=axis)
    return OpLib.add('GatherElements', inputs, axis=axis, **kwargs)


@OpSchema.num_inputs(1)
def identity(inputs, **kwargs):
    """Return a tensor copied from the input.

    Examples:

    ```python
    # Copy ``x`` to ``y``
    x = dragon.zeros(shape=(2, 3))
    y = dragon.identity(x)

    # ``x`` != ``y``
    x += 1
    print(x)
    print(y)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Identity', inputs)
    return OpLib.add('Identity', inputs, **kwargs)


@OpSchema.num_inputs(1)
def nonzero(inputs, **kwargs):
    r"""Return the index of non-zero elements.

    .. math:: \text{out} = \{i\}, \text{ if } \text{input}_{i} \neq 0

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('NonZero', inputs)
    return OpLib.add('NonZero', inputs, **kwargs)


@OpSchema.num_inputs(1)
def one_hot(inputs, depth, on_value=1, off_value=0, **kwargs):
    r"""Return the one-hot representation of input.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                \text{off\_value}, & \text{ if } \text{input}_{i} \neq j \\
                \text{on\_value}, & \text{ otherwise }
            \end{cases}

    The max value of input, i.e., the :attr:`depth` should be specified:

    ```python
    x = dragon.constant([0, 1, 2, 3])
    print(dragon.one_hot(x, depth=5))  # depth >= 4 will be ok
    ```

    Use :attr:`on_value` or :attr:`off_value` custom filling:

    ```python
    x = dragon.constant([0, 1, 2, 3])
    print(dragon.one_hot(x, depth=4, on_value=2, off_value=3))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    depth : int
        The depth of representation.
    on_value : number, optional, default=1
        The value for equal branch.
    off_value : number, optional, default=0
        The value for not-equal branch.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    on_value, off_value = float(on_value), float(off_value)
    if context.executing_eagerly():
        return OpLib.execute(
            'OneHot', inputs, depth=depth,
            on_value=on_value, off_value=off_value)
    return OpLib.add('OneHot', inputs, depth=depth, on_value=on_value,
                     off_value=off_value, **kwargs)


@OpSchema.num_inputs(1)
def pad(inputs, pads, mode='constant', value=0, **kwargs):
    """Pad the input according to the given sizes.

    :attr:`pads` should be a sequence with :math:`N` tuples,
    where :math:`N` is the number of input dimensions:

    ```python
    x = dragon.ones(shape=(2, 3))
    print(dragon.pad(x, [[0, 1], [1, 0]]))  # Ok, (2, 3) -> (3, 4)
    print(dragon.pad(x, [[0, 1]]))  # Wrong
    ```

    Following padding modes are supported:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])
    # ConstantPad
    print(dragon.pad(x, [[0, 1], [1, 0]], 'constant', 9))
    # ReflectPad
    print(dragon.pad(x, [[0, 1], [1, 0]], 'reflect'))
    # EdgePad
    print(dragon.pad(x, [[0, 1], [1, 0]], 'edge'))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    pads : Sequence[Tuple[int], Tuple[dragon.Tensor]]
        The padding begins and ends.
    mode : {'constant', 'reflect', 'edge'}, optional
        The padding mode.
    value : number, optional, default=0
        The value used in ``'constant'`` mode.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    value, mode = float(value), mode.upper()
    pads_begin, pads_end = [], []
    for v1, v2 in pads:
        pads_begin.append(v1)
        pads_end.append(v2)
    pads = pads_begin + pads_end
    if context.executing_eagerly():
        return OpLib.execute(
            'Pad', inputs, mode=mode, value=value,
            ndim=len(pads_begin), pads=pads)
    return OpLib.add('Pad', inputs, value=value, mode=mode, pads=pads, **kwargs)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg('repeats')
def repeat(inputs, axis=None, repeats=1, **kwargs):
    """Repeat the elements along the given axis.

    Examples:

    ```python
    x = dragon.constant([[1, 2], [3, 4]])

    # A negative axis is the last-k axis
    print(dragon.repeat(x, axis=1, repeats=2))  # [[1, 1, 2, 2], [3, 3, 4, 4]]
    print(dragon.repeat(x, axis=-1, repeats=2))  # Equivalent

    # If axis is None, repeat a flattened input
    print(dragon.repeat(x, repeats=2))  # [1, 1, 2, 2, 3, 3, 4, 4]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to repeat.
    repeats : Union[int, dragon.Tensor], optional, default=1
        The repeat size.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'Repeat', inputs, axis=axis, repeats=args['repeats'])
    return OpLib.add('Repeat', **args)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg(name='shape', name_v2='dims')
def reshape(inputs, shape, copy=True, **kwargs):
    """Change the dimensions of input.

    Examples:

    ```python
    # Provide a determined value for each dimension if possible
    x = dragon.ones(shape=(1, 2, 3, 4))
    print(dragon.reshape(x, shape=(6, 4)).shape)  # (6, 4)

    # Set the existing dimensions to ``0`` if it unchanged
    print(dragon.reshape(x, shape=(0, 0, 12)).shape)  # (1, 2, 12)
    print(dragon.reshape(x, shape=(0, 0, 0, 0)).shape)  # (1, 2, 3, 4)
    print(dragon.reshape(x, shape=(0, 0, 0, 0, 0)).shape)  # Wrong

    # You can also set ``-1`` once to infer the value
    print(dragon.reshape(x, shape=(-1, 4)).shape)  # (6, 4)
    print(dragon.reshape(x, shape=(-1, -1)).shape)  # Wrong
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    shape : Union[Sequence[int], dragon.Tensor]
        The output shape.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'Reshape', inputs, outputs=[None] if copy else inputs,
            ndim=len(args['dims']), dims=args['dims'])
    args.pop('copy')
    return OpLib.add('Reshape', **args)


@OpSchema.num_inputs(1)
def reverse(inputs, axis, **kwargs):
    """Reverse elements along the given axis.

    :attr:`axis` could be negative:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.reverse(x, axis=1))  # [[3, 2, 1], [6, 5, 4]]
    print(dragon.reverse(x, axis=-1))  # Equivalent

    # Also, axis could be a sequence of integers
    print(dragon.reverse(x, axis=(0, 1)))  # [[6, 5, 4], [3, 2, 1]]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]]
        The axis to reverse.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    axes = nest.flatten(axis) if axis is not None else axis
    if context.executing_eagerly():
        return OpLib.execute('Reverse', inputs, axes=axes)
    return OpLib.add('Reverse', inputs, axes=axes, **kwargs)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg('shift', name_v2='shifts')
def roll(inputs, shift, axis=None, **kwargs):
    """Roll elements along the given axis.

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.roll(x, shift=1, axis=1))  # [[3, 1, 2], [6, 4, 5]]
    print(dragon.roll(x, shift=1, axis=-1))  # Equivalent

    # If axis is None, roll input as a vector
    print(dragon.roll(x, shift=1))  # [[6, 1, 2], [3, 4, 5]]

    # Also, axis could be a sequence of integers
    print(dragon.roll(x, shift=(1, 1), axis=(0, 1)))  # [[6, 4, 5], [3, 1, 2]]
    print(dragon.roll(x, shift=(1, -1), axis=(0, 1)))  # [[5, 6, 4], [2, 3, 1]]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    shift : Union[int, Sequence[int], dragon.Tensor]
        The rolling offset of each axis.
    axis : Union[int, Sequence[int]], optional
        The axis to roll.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    axes = nest.flatten(axis) if axis is not None else axis
    if isinstance(shift, six.integer_types):
        args['shifts'] = nest.flatten(shift)
    if context.executing_eagerly():
        return OpLib.execute(
            'Roll', inputs, num_shifts=len(args['shifts']),
            shifts=args['shifts'], axes=axes)
    args.pop('axis')
    return OpLib.add('Roll', axes=axes, **args)


@OpSchema.num_inputs(3)
def scatter_add(inputs, axis=0, copy=True, **kwargs):
    """Add elements along the given axis of index.

    Number of dimensions of input and index should be same.
    For 3-d input, output is updated as:

    ```python
    out[index[i, j, k], j, k] += updates[i, j, k]  # ``axis`` is 0
    out[i, index[i, j, k], k] += updates[i, j, k]  # ``axis`` is 1
    out[i, j, index[i, j, k]] += updates[i, j, k]  # ``axis`` is 2
    ```

    Examples:

    ```python
    y = dragon.constant([[1, 2], [3, 4]])
    x = dragon.constant([[5, 6], [7, 8]])
    index = dragon.constant([[0, 0], [0, 0]])
    print(dragon.scatter_add([y, index, x], axis=0))  # [[13, 16], [3, 4]]
    print(dragon.scatter_add([y, index, x], axis=1))  # [[12, 2], [18, 4]]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input, index and updates tensor.
    axis : int, optional, default=0
        The axis of index values.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'ScatterAdd', inputs,
            outputs=[None] if copy else [inputs[0]], axis=axis)
    return OpLib.add('ScatterAdd', inputs, axis=axis, **kwargs)


@OpSchema.num_inputs(3)
def scatter_elements(inputs, axis=0, copy=True, **kwargs):
    """Update elements along the given axis of index.

    Number of dimensions of input and index should be same.
    For 3-d input, output is updated as:

    ```python
    out[index[i, j, k], j, k] = updates[i, j, k]  # ``axis`` is 0
    out[i, index[i, j, k], k] = updates[i, j, k]  # ``axis`` is 1
    out[i, j, index[i, j, k]] = updates[i, j, k]  # ``axis`` is 2
    ```

    Examples:

    ```python
    y = dragon.constant([[1, 2], [3, 4]])
    x = dragon.constant([[5, 6], [7, 8]])
    index = dragon.constant([[0, 0], [0, 1]])
    print(dragon.scatter_elements([y, index, x], axis=0))  # [[7, 6], [3, 8]]
    print(dragon.scatter_elements([y, index, x], axis=1))  # [[6, 2], [7, 8]]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input, index and updates tensor.
    axis : int, optional, default=0
        The axis of index values.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'ScatterElements', inputs,
            outputs=[None] if copy else [inputs[0]], axis=axis)
    return OpLib.add('ScatterElements', inputs, axis=axis, **kwargs)


@OpSchema.num_inputs(1)
def shape(inputs, **kwargs):
    """Return the shape of input.

    Examples:

    ```python
    x = dragon.ones((2, 3))
    print(x.shape)  # Return a sequence
    print(dragon.shape(x))  # Return a tensor
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The tensor shape.

    """
    if context.executing_eagerly():
        return OpLib.execute('Shape', inputs)
    return OpLib.add('Shape', inputs, **kwargs)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg('starts')
@OpSchema.convert_arg('sizes')
def slice(inputs, starts, sizes, **kwargs):
    """Select the elements according to the given sections.

    Each section should be hinted by a pair of ``[start, start + size)``:

    ```python
    x = dragon.constant([[[0, 1, 2], [3, 4, 5]]])
    print(dragon.slice(x, [0, 1, 2], [1, 1, 1]))  # [[[5]]]
    print(x[0:1, 1:2:, 2:3])  # Equivalent
    ```

    :attr:`sizes` accepts value ``-1`` or ``0``:

    ```python
    x = dragon.constant([[[0, 1, 2], [3, 4, 5]]])
    # Set ``0`` to squeeze dimensions with size 1
    print(dragon.slice(x, [0, 1, 2], [0, 0, 0]))  # 5
    # Set ``-1`` to take all the remained elements
    print(dragon.slice(x, [0, 0, 0], [-1, -1, -1]))  # [[[0, 1, 2], [3, 4, 5]]]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    starts : Union[Sequence[int], dragon.Tensor]
        The start location for each dimension.
    sizes : Union[Sequence[int], dragon.Tensor]
        The number of elements sliced from start.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'Slice', inputs, ndim=len(args['starts']),
            starts=args['starts'], sizes=args['sizes'])
    return OpLib.add('Slice', **args)


@OpSchema.num_inputs(1)
def split(inputs, num_or_size_splits, axis=0, copy=True, **kwargs):
    """Split input into chunks along the given axis.

    Either number or size of splits will be accepted:

    ```python
    x = dragon.constant([[1, 2], [3, 4], [5, 6]])
    # Shape: (3, 2) -> (2, 2), (1, 2)
    print(dragon.split(x, num_or_size_splits=2))
    # Shape: (3, 2) -> (1, 2), (2, 2)
    print(dragon.split(x, num_or_size_splits=(1, 2)))
    ```

    :attr:`axis` can be negative:

    ```python
    x = dragon.constant([[1, 2], [3, 4], [5, 6]])
    print(dragon.split(x, 2, axis=1))
    print(dragon.split(x, 2, axis=-1))  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    num_or_size_splits: Union[int, Sequence[int]]
        The number or size of chunks.
    axis : int, optional, default=0
        The axis to split.
    copy : bool, optional, default=True
        Copy or create the views of input.

    Returns
    -------
    Sequence[dragon.Tensor]
        The output tensors.

    """
    if nest.is_sequence(num_or_size_splits):
        size_splits = num_or_size_splits
        num = num_splits = len(num_or_size_splits)
    else:
        size_splits = None
        num, num_splits = num_or_size_splits, 0
    if context.executing_eagerly():
        return OpLib.execute(
            'Split', inputs, outputs=[None] * num, axis=axis,
            num_splits=num_splits, split=size_splits, copy=copy)
    return OpLib.add('Split', inputs, num_outputs=num, axis=axis,
                     split=size_splits, copy=copy, **kwargs)


@OpSchema.num_inputs(1)
def squeeze(inputs, axis=None, copy=True, **kwargs):
    """Remove the dimensions of input with size 1.

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.ones((1, 2, 2, 1))

    # Remove all matched dimensions if axis is None
    # Otherwise, only the specified axes will be removed
    print(dragon.squeeze(x).shape)          # (1, 2, 2, 1) -> (2, 2)
    print(dragon.squeeze(x, axis=0).shape)  # (1, 2, 2, 1) -> (2, 2, 1)

    # A negative axis is the last-k axis
    print(dragon.squeeze(x, axis=3).shape)   # (1, 2, 2, 1) -> (1, 2, 2)
    print(dragon.squeeze(x, axis=-1).shape)  # Equivalent

    # Also, axis could be a sequence of integers
    print(dragon.squeeze(x, axis=[0, 3]).shape)  # (1, 2, 2, 1) -> (2, 2)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to remove.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    axes = None if axis is None else nest.flatten(axis)
    if context.executing_eagerly():
        return OpLib.execute(
            'Squeeze', inputs, outputs=[None] if copy else inputs, axes=axes)
    return OpLib.add('Squeeze', inputs, axes=axes, **kwargs)


@OpSchema.num_inputs(1, 2147483647)
def stack(inputs, axis=0, **kwargs):
    """Stack the inputs along the given axis.

    All the dimensions of inputs should be same:

    ```python
    x1 = dragon.ones(shape=(2, 3))
    x2 = dragon.zeros(shape=(2, 4))
    y = dragon.stack([x1, x1])  # Ok
    z = dragon.stack([x1, x2])  # Wrong
    ```

    :attr:`axis` can be negative:

    ```python
    x = dragon.constant([[1, 2], [3, 4]])
    y = dragon.stack([x, x], axis=1)
    z = dragon.stack([x, x], axis=-1)  # Equivalent
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input tensors.
    axis : int, optional, default=0
        The axis to stack.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Stack', inputs, axis=axis)
    return OpLib.add('Stack', inputs, axis=axis, **kwargs)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg(name='repeats')
def tile(inputs, repeats, **kwargs):
    """Repeat elements along each axis of input.

    Examples:

    ```python
    x = dragon.constant([[1, 2], [3, 4]])
    print(dragon.tile(x, repeats=(1, 2)))  # [[1, 2, 1, 2], [3, 4, 3, 4]]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    repeats : Union[Sequence[int], dragon.Tensor]]
        The repetition for each axis.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'Tile', inputs, ndim=len(args['repeats']), repeats=args['repeats'])
    return OpLib.add('Tile', **args)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg('perm')
def transpose(inputs, perm=None, copy=True, **kwargs):
    """Permute the dimensions of input.

    Examples:

    ```python
    # Provide the permutation for all axes
    x = dragon.ones(shape=(2, 3, 4))
    print(dragon.transpose(x, (0, 2, 1)).shape)  # (2, 4, 3)

    # Or dimensions will be simply inverse
    print(dragon.transpose(x).shape)  # (4, 3, 2)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    perm : Union[Sequence[int], dragon.Tensor]], optional
        The output permutation.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'Transpose', inputs,
            outputs=[None] if copy else inputs,
            ndim=len(args['perm']) if perm is not None else 0,
            perm=args['perm'])
    return OpLib.add('Transpose', **args)


@OpSchema.num_inputs(1)
def tril(inputs, k=0, copy=True, **kwargs):
    r"""Return the lower triangular part of input.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                0, & \text{ if } j > i + k \\
                \text{input}_{ij}, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = dragon.ones((3, 3))
    print(dragon.tril(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    k : int, optional, default=0
        Diagonal above which to zero elements.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Trilu', inputs, outputs=[None] if copy else inputs,
            k=k, upper=False)
    return OpLib.add('Trilu', inputs, k=k, upper=False, **kwargs)


@OpSchema.num_inputs(1)
def triu(inputs, k=0, copy=True, **kwargs):
    r"""Return the upper triangular part of input.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                0, & \text{ if } j < i + k \\
                \text{input}_{ij}, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = dragon.ones((3, 3))
    print(dragon.triu(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    k : int, optional, default=0
        Diagonal below which to zero elements.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Trilu', inputs, outputs=[None] if copy else inputs,
            k=k, upper=True)
    return OpLib.add('Trilu', inputs, k=k, upper=True, **kwargs)


@OpSchema.num_inputs(1)
def unique(inputs, return_inverse=False, return_counts=False, **kwargs):
    """Return the unique elements of input.

    If ``return_inverse``, return the extra index where input mapping to:

    ```python
    x = dragon.constant([1, 2, 3, 2])
    y, index = dragon.unique(x, return_inverse=True)
    print(y)  # [1, 2, 3]
    print(index)  # [0, 1, 2, 1]
    ```

    If ``return_counts``, return the extra counts of output:

    ```python
    x = dragon.constant([1, 2, 3, 2])
    y, counts = dragon.unique(x, return_counts=True)
    print(y)  # [1, 2, 3]
    print(counts)  # [1, 2, 1]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    return_inverse : bool, optional, default=False
        Return the inverse index or not.
    return_counts : bool, optional, default=False
        Return the counts or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.
    dragon.Tensor, optional
        The inverse index tensor.
    dragon.Tensor, optional
        The counting tensor.

    """
    num_outputs = 1 + return_inverse + return_counts
    if context.executing_eagerly():
        return OpLib.execute(
            'Unique', inputs, outputs=[None] * num_outputs,
            return_inverse=return_inverse, return_counts=return_counts)
    return OpLib.add('Unique', inputs, num_outputs=num_outputs,
                     return_inverse=return_inverse,
                     return_counts=return_counts, **kwargs)


@OpSchema.num_inputs(1)
def unstack(inputs, axis=0, num=None, copy=True, **kwargs):
    """Unpack input into chunks along the given axis.

    The number of outputs should be equal to the dimension of axis:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])
    # Shape: (2, 3) -> (3,), (3,)
    print(dragon.unstack(x, axis=0))
    ```

    :attr:`axis` can be negative:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])
    # Shape: (2, 3) -> (2,), (2,), (2,)
    print(dragon.unstack(x, axis=-1))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=0
        The axis to unpack.
    num : int, optional
        The number of outputs.
    copy : bool, optional, default=True
        Copy or create the views of input.

    Returns
    -------
    Sequence[dragon.Tensor]
        The output tensors.

    """
    num_outputs = num or inputs[0].shape[axis]
    if context.executing_eagerly():
        return OpLib.execute(
            'Split', inputs, outputs=[None] * num_outputs,
            axis=axis, copy=copy, keepdims=False)
    return OpLib.add('Split', inputs, num_outputs=num_outputs,
                     axis=axis, copy=copy, keepdims=False, **kwargs)


@OpSchema.num_inputs(1, 3)
def where(inputs, **kwargs):
    r"""Select the elements from two branches under the condition.

    .. math::
        \text{out}_{i} =
            \begin{cases}
                \text{input1}_{i}, & \text{ if } \text{condition}_{i} \\
                \text{input2}_{i}, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    a = dragon.constant([1, 2, 3])
    b = dragon.constant([3, 2, 1])
    print(dragon.where([a > b, a, b]))  # [3, 2, 3]
    ```

    If only the ``condition`` is given,
    return the coordinates of ``True`` elements:

    ```python
    x = dragon.constant([[True, False, True],
                         [False, True, True]])
    print(dragon.where(x))  # [[0, 0], [0, 2], [1, 1], [1, 2]]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The condition, input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.nonzero(...)`_

    """
    if types.is_tensor(inputs) or len(inputs) == 1:
        return nonzero(inputs, **kwargs)
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute('Where', inputs)
    return OpLib.add('Where', inputs, **kwargs)
