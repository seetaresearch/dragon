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

from dragon.core.autograph.tensor import Tensor
from dragon.core.eager import context
from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import types
from dragon.core.ops import array_ops_lib
from dragon.core.ops.utils import ArgHelper
from dragon.core.ops.utils import OpSchema
from dragon.core.ops.utils import parse_args
from dragon.core.util import nest


@OpSchema.num_inputs(1)
def argmax(inputs, axis=None, keep_dims=False, **kwargs):
    """Compute the index of maximum elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative ``axis`` is the last-k axis
    print(dragon.math.argmax(x, 1))
    print(dragon.math.argmax(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar index
    print(dragon.math.argmax(x))  # 5
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to reduce.
    keep_dims : bool, optional, default=False
        Keep the reduced dimension or not.

    Returns
    -------
    dragon.Tensor
        The index of maximum elements.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.ArgReduce
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                op_type='ArgMax',
                axis=axis,
                keep_dims=keep_dims,
            ).apply([inputs])
    else:
        return op_lib.blend('ArgMax', **args)


@OpSchema.num_inputs(1)
def argmin(inputs, axis=None, keep_dims=False, **kwargs):
    """Compute the index of minimum elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative ``axis`` is the last-k axis
    print(dragon.math.argmin(x, 1))
    print(dragon.math.argmin(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar index
    print(dragon.math.argmin(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to reduce.
    keep_dims : bool, optional, default=False
        Keep the reduced dimension or not.

    Returns
    -------
    dragon.Tensor
        The index of minimum elements.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.ArgReduce
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                op_type='ArgMin',
                axis=axis,
                keep_dims=keep_dims,
            ).apply([inputs])
    else:
        return op_lib.blend('ArgMin', **args)


@OpSchema.num_inputs(1)
@ArgHelper.repeated_desc(name='shape', name_v2='dims')
def broadcast_to(inputs, shape, **kwargs):
    """Broadcast input according to a given shape.

    The length of ``shape`` could either be less or
    more than the number of input dimensions:

    ```python
    a = dragon.constant([[1], [2], [3]])
    # Shape: (3, 1) -> (3, 2)
    print(dragon.broadcast_to(a, shape=(3, 2)))
    print(dragon.broadcast_to(a, shape=(2,)))     # Equivalent
    print(dragon.broadcast_to(a, shape=(-1, 2)))  # Equivalent

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
    args = parse_args(locals())
    op_lib = array_ops_lib.Expand
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(args['dims']),
            ).apply([inputs], args['dims'])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def cast(inputs, dtype, **kwargs):
    """Cast the data type of input.

    Examples:

    ```python
    x = dragon.constant([1, 2, 3], dtype='int64')
    print(dragon.cast(x, dtype='float32'))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    dtype : str
        The data type to cast to.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = array_ops_lib.Cast
    if context.executing_eagerly():
        return op_lib \
            .instantiate(dtype=dtype) \
            .apply([inputs], inplace=inplace)
    else:
        if inputs.dtype == dtype:
            return inputs
        if inplace:
            args['inputs'], args['outputs'] = [], [inputs]
        return op_lib.blend(**args)


@OpSchema.num_inputs(2, 3)
def channel_affine(inputs, axis=1, num_axes=1, **kwargs):
    r"""Apply affine transformation along the channels.

    .. math:: \text{out} = \text{weight} * \text{input} + \text{bias}

    The range of channels to transform is given by:

    .. math:: [\text{axis}, \text{axis} + \text{num\_axes})

    Set ``axis`` to specific the start axis.

    Set ``num_axes`` to -1 will transform all remained axes.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input, weight and optional bias tensor.
    axis : int, optional, default=1
        The start axis, can be negative.
    num_axes : int, optional, default=1
        The number of axes to transform.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = array_ops_lib.ChannelAffine
    if context.executing_eagerly():
        return op_lib \
            .instantiate(axis=axis, num_axes=num_axes) \
            .apply(inputs, inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
@ArgHelper.repeated_desc('perm')
def channel_normalize(
    inputs,
    mean,
    std,
    axis=-1,
    dtype='float32',
    perm=None,
    **kwargs
):
    """Normalize channels with mean and standard deviation.

    The ``axis`` can be negative representing the last-k axis:

    ```python
    m = s = (1., 1., 1.)
    x = dragon.constant([1, 2, 3])
    print(dragon.channel_normalize(x, m, s, axis=0))   # [0., 1., 2.]
    print(dragon.channel_normalize(x, m, s, axis=-1))  # Equivalent
    ```

    If ``perm`` is provided, ``axis`` is selected from the output layout:

    ```python
    m, s = (1., 2., 3.), (1., 1., 1.)
    x = dragon.constant([[1, 2, 3]])
    # Provided 3 values to normalize the last axis
    # with length 1, only the first value will be taken
    print(dragon.channel_normalize(x, m, s, perm=(1, 0)))  # [[0.], [1.], [2.]]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    mean : Sequence[float], required
        The mean to subtract.
    std : Sequence[float], required
        The standard deviation to divide.
    axis : int, optional, default=-1
        The axis to normalize.
    dtype : str, optional, default='float32'
        The output data type.
    perm : Sequence[Union[int, dragon.Tensor]], optional
        The output permutation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.ChannelNormalize
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                ndim=len(args['perm']) if perm is not None else 0,
                mean=mean,
                std=std,
                dtype=dtype,
            ).apply([inputs], args['perm'])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def channel_shuffle(inputs, axis=0, group=1, **kwargs):
    """Shuffle channels between a given number of groups.
    `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

    Parameters
    ----------
    inputs : dragon.Tensor
        The inputs.
    axis : int, optional, default=0
        The axis of channels.
    group : int, optional, default=1
        The number of groups.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.ChannelShuffle
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                group=group,
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1, 2147483647)
def concat(inputs, axis=0, **kwargs):
    """Concatenate the inputs along the given axis.

    All dimensions except the ``axis`` should be same:

    ```python
    x1 = dragon.ones(shape=(2, 3))
    x2 = dragon.zeros(shape=(2, 4))
    y = dragon.concat([x1, x2], axis=1)  # Ok
    z = dragon.concat([x1, x2], axis=0)  # Wrong
    ```

    The ``axis`` can be negative representing the last-k axis:

    ```python
    y = dragon.concat([x1, x2], axis=1)
    z = dragon.concat([x1, x2], axis=-1)  # z == y
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
    args = parse_args(locals())
    op_lib = array_ops_lib.Concat
    if context.executing_eagerly():
        return op_lib.instantiate(axis=axis).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def cumsum(inputs, axis=0, exclusive=False, reverse=False, **kwargs):
    """Compute the cumulative sum of elements along the given axis.

    The argument ``axis`` could be negative:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.math.cumsum(x, 1))   # [[1, 3, 6], [4, 9, 15]]
    print(dragon.math.cumsum(x, -1))  # Equivalent
    ```

    To exclude the top element, set the ``exclusive``:

    ```python
    x = dragon.constant([1, 2, 3])
    print(dragon.math.cumsum(x, exclusive=True))  # [0, 1, 3]
    ```

    Also, ``reverse`` could be set to reverse the cumulative direction:

    ```python
    x = dragon.constant([1, 2, 3])
    print(dragon.math.cumsum(x))  # [1, 3, 6]
    print(dragon.math.cumsum(x, reverse=True))  # [6, 5, 3]
    print(dragon.math.cumsum(x, exclusive=True, reverse=True))  # [5, 3, 0]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=0
        The cumulative axis.
    exclusive : bool, optional, default=False
        **True** to exclude the top element.
    reverse : bool, optional, default=False
        **True** to compute in the reverse direction.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.Cumulative
    if context.executing_eagerly():
        return op_lib  \
            .instantiate(
                operation='Sum',
                axis=axis,
                exclusive=exclusive,
                reverse=reverse,
            ).apply([inputs])
    else:
        return op_lib.blend('CumSum', **args)


@OpSchema.num_inputs(1)
def expand_dims(inputs, axis, **kwargs):
    """Expand the dimensions of input with size 1.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.ones((2, 3, 4, 5))

    # ``axis`` determines the size-1 position in output
    print(dragon.expand_dims(x, axis=0).shape)  # (2, 3, 4, 5) -> (1, 2, 3, 4, 5)
    print(dragon.expand_dims(x, axis=1).shape)  # (2, 3, 4, 5) -> (2, 1, 3, 4, 5)

    # A negative axis is the last-k axis
    print(dragon.expand_dims(x, axis=4).shape)   # (2, 3, 4, 5) -> (2, 3, 4, 5, 1)
    print(dragon.expand_dims(x, axis=-1).shape)  # Equivalent

    # Also, ``axis`` could be a sequence of integers
    print(dragon.expand_dims(x, axis=[-1, -3]).shape)  # (2, 3, 4, 5) -> (2, 3, 4, 1, 5, 1)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]]
        The axis to insert the new dimension(s).

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args.pop('axis')
    args['axes'] = None if axis is None else nest.flatten(axis)
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = array_ops_lib.ExpandDims
    if context.executing_eagerly():
        return op_lib.instantiate(
            axes=args['axes'],
        ).apply([inputs], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def flatten(inputs, axis=0, num_axes=-1, keep_axes=None, **kwargs):
    r"""Flatten the input along the given axes.

    Set ``keep_axes`` to flatten if shape is dynamic.

    Examples:

    ```python
    x = dragon.Tensor(shape=[1, 2, 3, 4]).constant()
    print(dragon.flatten(x, axis=1, num_axes=-1).shape)  # (1, 24)
    print(dragon.flatten(x, axis=1, num_axes=2).shape)  # (1, 6, 4)
    print(dragon.flatten(x, keep_axes=1))  # (24,)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=0
        The start axis to flatten, can be negative.
    num_axes : int, optional, default=-1
        The number of axes to flatten.
    keep_axes : int, optional
        The number of axes to keep.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = array_ops_lib.Flatten
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                num_axes=num_axes,
                keep_axes=keep_axes,
            ).apply([inputs], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def index_select(inputs, index, axis=0, **kwargs):
    """Select the elements according to the index along the given axis.

    ``index`` could be a **int64** tensor or a sequence with integers:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])
    print(dragon.index_select(x, [0, 1]))
    print(dragon.index_select(x, dragon.constant([0, 1], 'int64')))
    ```

    More than one axis could be specified for ``index``:

    ```python
    # The number of ``axis`` should less than ``rank(index)``
    # And these axes should be continuous
    print(dragon.index_select(x, [0, 1], axis=[0, 1]))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    index : Union[Sequence[int], dragon.Tensor]
        The index to select elements.
    axis : Union[int, Sequence[int]], optional, default=0
        The axis where the index aligned.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    axes = nest.flatten(axis)
    axes.sort()
    if axes[-1] != (axes[0] + len(axes) - 1):
        raise ValueError('The <axis> should be a continuous sequence.')
    op_lib = array_ops_lib.IndexSelect
    if context.executing_eagerly():
        if not types.is_eager_tensor(index):
            index = EagerTensor(index, dtype='int64')
        return op_lib \
            .instantiate(
                axis=axes[0],
                num_axes=len(axes),
            ).apply([inputs, index])
    else:
        if not isinstance(index, Tensor):
            index = Tensor.convert_to(index, 'int64')
        args['inputs'], args['index'] = \
            [args['inputs'], index], None
        args['axis'], args['num_axes'] = axes[0], len(axes)
        return op_lib.blend(**args)


@OpSchema.num_inputs(2)
def masked_select(inputs, **kwargs):
    """Select the elements of input where mask is 1.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input and mask tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.MaskedSelect
    if context.executing_eagerly():
        return op_lib.instantiate().apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def max(inputs, axis=None, keep_dims=False, **kwargs):
    """Compute the max value of elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.math.max(x, 1))
    print(dragon.math.max(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.max(x))  # Result is 6

    # Also, ``axis`` could be a sequence of integers
    print(dragon.math.max(x, [0, 1]))  # Result is 6
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to reduce.
    keep_dims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args.pop('axis')
    args['axes'] = None if axis is None else nest.flatten(axis)
    op_lib = array_ops_lib.Reduce
    if context.executing_eagerly():
        return op_lib  \
            .instantiate(
                operation='Max',
                axes=args['axes'],
                keep_dims=keep_dims,
            ).apply([inputs])
    else:
        return op_lib.blend('ReduceMax', **args)


@OpSchema.num_inputs(1)
def mean(inputs, axis=None, keep_dims=False, **kwargs):
    """Compute the mean value of elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.math.mean(x, 1))
    print(dragon.math.mean(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.mean(x))  # Result is 3

    # Also, ``axis`` could be a sequence of integers
    print(dragon.math.mean(x, [0, 1]))  # Result is 3
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keep_dims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args.pop('axis')
    args['axes'] = None if axis is None else nest.flatten(axis)
    op_lib = array_ops_lib.Reduce
    if context.executing_eagerly():
        return op_lib  \
            .instantiate(
                operation='Mean',
                axes=args['axes'],
                keep_dims=keep_dims,
            ).apply([inputs])
    else:
        return op_lib.blend('ReduceMean', **args)


@OpSchema.num_inputs(1)
def min(inputs, axis=None, keep_dims=False, **kwargs):
    """Compute the min value of elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.math.min(x, 1))
    print(dragon.math.min(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.min(x))  # Result is 1

    # Also, ``axis`` could be a sequence of integers
    print(dragon.math.min(x, [0, 1]))  # Result is 1
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to reduce.
    keep_dims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args.pop('axis')
    args['axes'] = None if axis is None else nest.flatten(axis)
    op_lib = array_ops_lib.Reduce
    if context.executing_eagerly():
        return op_lib  \
            .instantiate(
                operation='Min',
                axes=args['axes'],
                keep_dims=keep_dims,
            ).apply([inputs])
    else:
        return op_lib.blend('ReduceMin', **args)


@OpSchema.num_inputs(1)
def moments(inputs, axis=None, keep_dims=False, **kwargs):
    r"""Compute the mean and variance of input along the given axes.

    .. math::
        \begin{cases}
            \text{mean} = \frac{1}{n}\sum(\text{input}) \\
            \text{variance} = \frac{1}{n}\sum(x - \text{mean}(\text{input}))^{2}
        \end{cases}

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.math.moments(x, 1))
    print(dragon.math.moments(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.moments(x))  # Mean is 3.5, Var is 2.916667

    # Also, ``axis`` could be a sequence of integers
    print(dragon.math.moments(x, [0, 1]))  # Mean is 3.5, Var is 2.916667
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor :math:`x`.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keep_dims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The mean tensor.
    dragon.Tensor
        The variance tensor.

    """
    args = parse_args(locals())
    args.pop('axis')
    args['axes'] = None if axis is None else nest.flatten(axis)
    op_lib = array_ops_lib.Moments
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axes=args['axes'],
                keep_dims=keep_dims,
            ).apply([inputs])
    else:
        return op_lib.blend(num_outputs=2, **args)


@OpSchema.num_inputs(1)
def multinomial(inputs, num_samples=1, epsilon=0, normalize=False, **kwargs):
    """Return a tensor with index sampled from multinomial distribution.

    If ``normalize`` is **True**, negative input is accepted.

    Otherwise, input should be non-negative.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    num_samples : int, optional, default=1
        The number of samples.
    epsilon : float, optional, default=0
        The epsilon value to apply e-greedy strategy.
    normalize : bool, optional, default=False
        Whether to normalize the input.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['epsilon'] = float(epsilon)
    op_lib = array_ops_lib.Multinomial
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                num_samples=num_samples,
                epsilon=args['epsilon'],
                normalize=normalize,
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


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
    args = parse_args(locals())
    op_lib = array_ops_lib.NonZero
    if context.executing_eagerly():
        return op_lib.instantiate().apply([inputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def one_hot(inputs, depth, on_value=1, off_value=0, **kwargs):
    r"""Return the one-hot representation for input.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                \text{off\_value}, & \text{ if } \text{input}_{i} \neq j \\
                \text{on\_value}, & \text{ otherwise }
            \end{cases}

    The max value of input, i.e., the ``depth`` should be specified:

    ```python
    x = dragon.constant([0, 1, 2, 3], dtype='int64')
    print(dragon.one_hot(x, depth=5))  # depth >= 4 will be ok
    ```

    You can also set the ``on_value`` or ``off_value``:

    ```python
    print(dragon.one_hot(x, depth=4, on_value=2, off_value=3))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    depth : int
        The depth of representation.
    on_value : int, optional, default=1
        The value for equal branch.
    off_value : int, optional, default=0
        The value for not-equal branch.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.OneHot
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                depth=depth,
                on_value=on_value,
                off_value=off_value,
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def pad(inputs, pads, mode='constant', value=0, **kwargs):
    r"""Pad the input according to the given sizes.

    The ``pads`` should be a sequence with :math:`N` tuples,
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
        The begins and ends of padding.
    mode : {'constant', 'reflect', 'edge'}, optional
        The padding mode.
    value : number, optional, default=0
        The value used in **constant** mode.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['value'] = float(value)
    args['mode'] = mode.upper()
    pads_begin, pads_end = [], []
    for pad in pads:
        if len(pad) != 2:
            raise ValueError(
                'The tuple length of <pads> '
                'should be 2, got {}.'.format(len(pad)))
        pads_begin.append(pad[0])
        pads_end.append(pad[1])
    args['pads'] = pads_begin + pads_end
    op_lib = array_ops_lib.Pad
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(pads_begin),
                value=args['value'],
                mode=args['mode'],
            ).apply([inputs], args['pads'])
    else:
        return op_lib.blend(**args)


def permutation(limit, dtype='int64', **kwargs):
    r"""Return a tensor with value in the permuted range.

    Specify ``limit`` to determine an interval :math:`[0, \text{limit})`:

    ```python
    x = dragon.permutation(4)
    ```

    Parameters
    ----------
    limit: number
        The end of interval.
    dtype : str, optional, default='int64'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['dtype'] = args['dtype'].lower()
    op_lib = array_ops_lib.Permutation
    trainable = args.pop('trainable') if 'trainable' in args else False
    if context.executing_eagerly():
        return op_lib \
            .instantiate(dtype=dtype) \
            .apply(limit, trainable=trainable)
    else:
        return op_lib.blend(**args)


def range(start, limit=None, delta=1, dtype='int64', **kwargs):
    r"""Return a tensor of evenly spaced values within a interval.

    Specify ``start`` and ``limit`` to determine an interval:

    ```python
    x = dragon.range(2, 4)  # [2, 3]
    ```

    If ``limit`` is **None**, interval :math:`[0, \text{start})` will be taken instead:

    ```python
    x = dragon.range(5)  # [0, 1, 2, 3, 4]
    ```

    Set ``delta`` to make the strides:

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
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['dtype'] = args['dtype'].lower()
    if limit is None:
        args['slice'] = (float(start), float(delta))
    else:
        args['slice'] = (float(start), float(limit), float(delta))
    args.pop('start')
    args.pop('limit')
    args.pop('delta')
    op_lib = array_ops_lib.Range
    trainable = args.pop('trainable') if 'trainable' in args else False
    if context.executing_eagerly():
        return op_lib.instantiate(
            num_args=len(args['slice']),
            dtype=dtype,
        ).apply(args['slice'], trainable=trainable)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
@ArgHelper.desc('repeats')
def repeat(inputs, axis=None, repeats=1, **kwargs):
    """Repeat the elements along the given axis.

    If ``axis`` is **None**, flattened results will be returned.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to repeat.
    repeats : Union[int, dragon.Tensor], optional, default=1
        The magnitude of repeating.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.Repeat
    if context.executing_eagerly():
        return op_lib  \
            .instantiate(
                axis=axis,
                repeats=args['repeats'],
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
@ArgHelper.repeated_desc(name='shape', name_v2='dims')
def reshape(inputs, shape, **kwargs):
    r"""Change the dimensions of input.

    Examples:

    ```python
    # Provide a determined value for each dimension if possible
    x = dragon.ones(shape=(1, 2, 3, 4))
    print(dragon.reshape(x, shape=[6, 4]).shape)  # [6, 4]

    # Set the existing dimensions to ``0`` if it unchanged
    print(dragon.reshape(x, shape=[0, 0, 12]).shape)  # [1, 2, 12]
    print(dragon.reshape(x, shape=[0, 0, 0, 0]).shape)  # [1, 2, 3, 4]
    print(dragon.reshape(x, shape=[0, 0, 0, 0, 0]).shape)  # Wrong

    # You can also set ``-1`` once to infer the value
    print(dragon.reshape(x, shape=[-1, 4]).shape)  # [6, 4]
    print(dragon.reshape(x, shape=[-1, -1]).shape)  # Wrong
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    shape : Union[Sequence[int], dragon.Tensor]
        The output shape.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = array_ops_lib.Reshape
    if context.executing_eagerly():
        return op_lib \
            .instantiate(ndim=len(args['dims'])) \
            .apply([inputs], args['dims'], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def shape(inputs, **kwargs):
    r"""Return the shape of input.

    Examples:

    ```python
    x = dragon.ones((2, 3))
    print(x.shape)          # Return a sequence
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
    args = parse_args(locals())
    op_lib = array_ops_lib.Shape
    if isinstance(inputs, EagerTensor):
        return op_lib.instantiate().apply([inputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
@ArgHelper.repeated_desc('starts')
@ArgHelper.repeated_desc('sizes')
def slice(inputs, starts, sizes, **kwargs):
    """Select the elements according to the given sections.

    Each section should be hinted by a pair of ``[start, start + size)``:

    ```python
    x = dragon.constant([[[0, 1, 2], [3, 4, 5]]])
    print(dragon.slice(x, [0, 1, 2], [1, 1, 1]))  # [[[5]]]
    print(x[0:1, 1:2:, 2:3])  # Equivalent
    ```

    The ``sizes`` accepts value **-1** or **0**:

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
    args = parse_args(locals())
    op_lib = array_ops_lib.Slice
    if context.executing_eagerly():
        return op_lib \
            .instantiate(ndim=len(args['starts'])) \
            .apply([inputs], args['starts'], args['sizes'])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def split(
    inputs,
    num_or_size_splits,
    axis=0,
    slice_points=None,
    **kwargs
):
    r"""Split input into chunks along the given axis.

    Either number or size of splits will be accepted:

    ```python
    x = dragon.constant([[1, 2], [3, 4], [5, 6]])
    # Shape: (3, 2) -> (2, 2), (1, 2)
    print(dragon.split(x, num_or_size_splits=2))
    # Shape: (3, 2) -> (1, 2), (2, 2)
    print(dragon.split(x, num_or_size_splits=(1, 2)))
    ```

    The ``axis`` can be negative representing the last-k axis:

    ```python
    x = dragon.constant([[1, 2], [3, 4], [5, 6]])
    print(dragon.split(x, 2, axis=1))
    print(dragon.split(x, 2, axis=-1))  # Equivalent
    ```

    Optionally, use ``slice_points`` to hint the splits:

    ```python
    x = dragon.constant([[1, 2], [3, 4], [5, 6]])
    # Shape: (3, 2) -> (1, 2), (1, 2), (1, 2)
    print(dragon.split(x, 3, slice_points=[1, 2]))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    num_or_size_splits: Union[int, Sequence[int]]
        The number or size of chunks.
    axis : int, optional, default=0
        The axis to split, can be negative.
    slice_points : Sequence[int], optional
        The optional slice points.

    Returns
    -------
    Sequence[dragon.Tensor]
        The outputs.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.Split
    if nest.is_sequence(num_or_size_splits):
        num_splits = len(num_or_size_splits)
        size_splits = num_or_size_splits
    else:
        num_splits = num_or_size_splits
        size_splits = None
    if slice_points is not None:
        if len(slice_points) + 1 != num_splits:
            raise ValueError(
                'Excepted %d values for <slice_points>.'
                % len(slice_points))
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                axis=axis,
                size_splits=size_splits,
                slice_points=slice_points,
            ).apply([inputs], num_splits)
    else:
        args.pop('num_or_size_splits')
        args['size_splits'] = size_splits
        return op_lib.blend(num_outputs=num_splits, **args)


@OpSchema.num_inputs(1)
def squeeze(inputs, axis=None, **kwargs):
    """Remove the dimensions of input with size 1.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.ones((1, 2, 2, 1))

    # Remove all matched dimensions if ``axis`` is None
    # Otherwise, only the specified axes will be removed
    print(dragon.squeeze(x).shape)          # (1, 2, 2, 1) -> (2, 2)
    print(dragon.squeeze(x, axis=0).shape)  # (1, 2, 2, 1) -> (2, 2, 1)

    # A negative axis is the last-k axis
    print(dragon.squeeze(x, axis=3).shape)   # (1, 2, 2, 1) -> (1, 2, 2)
    print(dragon.squeeze(x, axis=-1).shape)  # Equivalent

    # Also, ``axis`` could be a sequence of integers
    print(dragon.squeeze(x, axis=[0, 3]).shape)  # (1, 2, 2, 1) -> (2, 2)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to remove.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args.pop('axis')
    args['axes'] = None if axis is None else nest.flatten(axis)
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = array_ops_lib.Squeeze
    if context.executing_eagerly():
        return op_lib \
            .instantiate(axes=args['axes']) \
            .apply([inputs], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1, 2147483647)
def stack(inputs, axis=0, **kwargs):
    """Stack the inputs along the given axis.

    All the dimensions of inputs should be same.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The inputs.
    axis : int, optional, default=0
        The axis to stack, can be negative.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.Stack
    if context.executing_eagerly():
        return op_lib \
            .instantiate(axis=axis) \
            .apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def sum(inputs, axis=None, keep_dims=False, **kwargs):
    """Compute the sum value of elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.math.sum(x, 1))
    print(dragon.math.sum(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.sum(x))  # Result is 21

    # Also, ``axis`` could be a sequence of int
    print(dragon.math.sum(x, [0, 1]))  # Result is 21
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keep_dims : bool, optional
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args.pop('axis')
    args['axes'] = None if axis is None else nest.flatten(axis)
    op_lib = array_ops_lib.Reduce
    if context.executing_eagerly():
        return op_lib  \
            .instantiate(
                operation='Sum',
                axes=args['axes'],
                keep_dims=keep_dims,
            ).apply([inputs])
    else:
        return op_lib.blend('ReduceSum', **args)


@OpSchema.num_inputs(1)
@ArgHelper.repeated_desc(name='repeats')
def tile(inputs, repeats, **kwargs):
    r"""Tile the input according to the given repeats.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    repeats : Sequence[Union[int, dragon.Tensor]]
        The number of repetitions for each axis.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.Tile
    if context.executing_eagerly():
        return op_lib \
            .instantiate(ndim=len(args['repeats'])) \
            .apply([inputs], args['repeats'])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
@ArgHelper.repeated_desc('perm')
def transpose(inputs, perm=None, **kwargs):
    r"""Permute the dimensions of input.

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
    perm : Sequence[Union[int, dragon.Tensor]], optional
        The output permutation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.Transpose
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(args['perm']) if perm is not None else 0,
            ).apply([inputs], args['perm'])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def top_k(inputs, k=1, axis=None, largest=True, sorted=True, **kwargs):
    """Return the top-K largest or smallest elements along the given axis.

    If ``axis`` is not given, the last axis is chosen:

    ```python
    x = dragon.constant([[1, 2, 3], [3, 2, 1]])
    value1, index1 = dragon.math.top_k(x, k=2)
    value2, index2 = dragon.math.top_k(x, k=2, axis=1)  # Equivalent
    ```

    If ``largest`` is **False**, the k smallest elements are returned:

    ```python
    x = dragon.constant([[1, 2, 3], [3, 2, 1]])
    _, index1 = dragon.math.top_k(x, largest=False)
    _, index2 = dragon.math.top_k(-x, largest=True)  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    k : int, optional, default=1
        The number of top elements to select.
    axis : int, optional
        The axis to reduce.
    largest : bool, optional, default=True
        Return largest or smallest elements.
    sorted : bool, optional
        Whether to return in the sorted order.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The value and index tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.TopK
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                k=k,
                axis=axis,
                largest=largest,
                sorted=sorted,
            ).apply([inputs])
    else:
        args['num_outputs'] = 2
        return op_lib.blend(**args)


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
        The counts tensor.

    """
    args = parse_args(locals())
    op_lib = array_ops_lib.Unique
    if context.executing_eagerly():
        return op_lib.instantiate(
            return_inverse=return_inverse,
            return_counts=return_counts,
        ).apply([inputs])
    else:
        num_outputs = 1 + return_inverse + return_counts
        return op_lib.blend(num_outputs=num_outputs, **args)


@OpSchema.num_inputs(1, 3)
def where(inputs, **kwargs):
    r"""Select the elements from two branches under the condition.

    .. math::
        \text{out}_{i} =
            \begin{cases}
                \text{input1}_{i}, & \text{ if } \text{condition}_{i} \\
                \text{input2}_{i}, & \text{ otherwise }
            \end{cases}

    Return the index of **True** elements, if only the ``condition`` is given.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1, input2 and condition tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    dragon.nonzero()

    """
    if types.is_tensor(inputs) or len(inputs) == 1:
        return nonzero(inputs, **kwargs)
    args = parse_args(locals())
    op_lib = array_ops_lib.Where
    if context.executing_eagerly():
        return op_lib.instantiate().apply(inputs)
    else:
        return op_lib.blend(**args)
