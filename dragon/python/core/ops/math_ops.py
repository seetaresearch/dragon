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
"""Math ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema
from dragon.core.ops import constant_ops
from dragon.core.util import nest


@OpSchema.num_inputs(1)
def abs(inputs, **kwargs):
    r"""Compute the absolute value of input.

    .. math:: \text{out} = \left| \text{input} \right|

    Examples:

    ```python
    x = dragon.constant([-1, 0, 1])
    print(dragon.math.abs(x))  # [1, 0, 1]
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
        return OpLib.execute('Abs', inputs)
    return OpLib.add('Abs', inputs, **kwargs)


@OpSchema.num_inputs(2)
def add(inputs, **kwargs):
    r"""Compute the element-wise addition.

    .. math:: \text{out} = \text{input1} + \text{input2}

    Examples:

    ```python
    a = dragon.constant(1)
    b = dragon.constant(2)
    print(dragon.math.add([a, b]))
    print(a + b)  # Equivalent operation
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Add', inputs)
    return OpLib.add('Add', inputs, **kwargs)


@OpSchema.num_inputs(2, 3)
def affine(inputs, axis=-1, **kwargs):
    """Apply affine transformation to input.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input, scale and bias tensor.
    axis : Union[int, Sequence[int]], optional, default=-1
        The axis to apply.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    axes = nest.flatten(axis)
    outputs = kwargs.pop('outputs', [None])
    if context.executing_eagerly():
        return OpLib.execute('Affine', inputs, outputs=outputs, axes=axes)
    return OpLib.add('Affine', inputs, axes=axes, **kwargs)


@OpSchema.num_inputs(1)
def argmax(inputs, axis=0, keepdims=False, **kwargs):
    """Compute the index of maximum elements along the given axis.

    :attr:`axis` could be negative:

    ```python
    # A negative axis is the last-k axis
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])
    print(dragon.math.argmax(x, axis=1))
    print(dragon.math.argmax(x, axis=-1))  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=0
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimension or not.

    Returns
    -------
    dragon.Tensor
        The index of maximum elements.

    """
    if context.executing_eagerly():
        return OpLib.execute('ArgMax', inputs, axis=axis, keepdims=keepdims)
    return OpLib.add('ArgMax', inputs, axis=axis, keepdims=keepdims, **kwargs)


@OpSchema.num_inputs(1)
def argmin(inputs, axis=0, keepdims=False, **kwargs):
    """Compute the index of minimum elements along the given axis.

    :attr:`axis` could be negative:

    ```python
    # A negative axis is the last-k axis
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])
    print(dragon.math.argmin(x, axis=1))
    print(dragon.math.argmin(x, axis=-1))  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=0
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimension or not.

    Returns
    -------
    dragon.Tensor
        The index of minimum elements.

    """
    if context.executing_eagerly():
        return OpLib.execute('ArgMin', inputs, axis=axis, keepdims=keepdims)
    return OpLib.add('ArgMin', inputs, axis=axis, keepdims=keepdims, **kwargs)


@OpSchema.num_inputs(2)
def atan2(inputs, **kwargs):
    r"""Compute the element-wise arc-tangent of two arguments.

    .. math:: \text{out} = \text{arctan}(\frac{\text{input1}}{\text{input2}})

    Examples:

    ```python
    y = dragon.constant(1)
    x = dragon.constant(2)
    print(dragon.math.atan2([y, x]))  # 0.46364761
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Atan2', inputs)
    return OpLib.add('Atan2', inputs, **kwargs)


@OpSchema.num_inputs(2)
def bitwise_and(inputs, **kwargs):
    r"""Compute the element-wise AND bitwise operation.

    .. math:: \text{out} = \text{input1} \mathbin{\&} \text{input2}

    Examples:

    ```python
    a = dragon.constant([0, -1, 2, -3, 4])
    b = dragon.constant([-4, 3, -2, 1, 0])
    print(dragon.bitwise.bitwise_and([a, b]))  # [0, 3, 2, 1, 0]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('BitwiseAnd', inputs)
    return OpLib.add('BitwiseAnd', inputs, **kwargs)


@OpSchema.num_inputs(2)
def bitwise_or(inputs, **kwargs):
    r"""Compute the element-wise OR bitwise operation.

    .. math:: \text{out} = \text{input1} \mathbin{|} \text{input2}

    Examples:

    ```python
    a = dragon.constant([0, -1, 2, -3, 4])
    b = dragon.constant([-4, 3, -2, 1, 0])
    print(dragon.bitwise.bitwise_or([a, b]))  # [-4, -1, -2, -3, 4]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('BitwiseOr', inputs)
    return OpLib.add('BitwiseOr', inputs, **kwargs)


@OpSchema.num_inputs(2)
def bitwise_xor(inputs, **kwargs):
    r"""Compute the element-wise XOR bitwise operation.

    .. math:: \text{out} = \text{input1} \oplus \text{input2}

    Examples:

    ```python
    a = dragon.constant([0, -1, 2, -3, 4])
    b = dragon.constant([-4, 3, -2, 1, 0])
    print(dragon.bitwise.bitwise_xor([a, b]))  # [-4, -4, -4, -4, 4]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('BitwiseXor', inputs)
    return OpLib.add('BitwiseXor', inputs, **kwargs)


@OpSchema.num_inputs(1)
def cast(inputs, dtype, copy=True, **kwargs):
    """Convert the data type of input.

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
        The data type to convert to.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Cast', inputs, outputs=[None] if copy else inputs, dtype=dtype)
    return OpLib.add('Cast', inputs, dtype=dtype, **kwargs)


@OpSchema.num_inputs(1)
def ceil(inputs, **kwargs):
    r"""Compute the smallest integer not less than input.

    .. math:: \text{out} = \lceil \text{input} \rceil

    Examples:

    ```python
    x = dragon.constant([1.4, 1.7, 2.0])
    print(dragon.math.ceil(x))  # [2., 2., 2.]
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
        return OpLib.execute('Ceil', inputs)
    return OpLib.add('Ceil', inputs, **kwargs)


@OpSchema.num_inputs(1)
def clip(inputs, low=None, high=None, **kwargs):
    r"""Compute the clipped input according to the given bounds.

    .. math:: \text{out} = \min(\max(\text{input}, \text{low}), \text{high})

    Examples:

    ```python
    x = dragon.constant([0, 1, 2, 3])
    print(dragon.math.clip(x, low=1))  # [1, 1, 2, 3]
    print(dragon.math.clip(x, high=2))  # [0, 1, 2, 2]
    print(dragon.math.clip(x, low=1, high=2))  # [1, 1, 2, 2]
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    low : number, optional
        The value to :math:`\text{low}`.
    high : number, optional
        The value to :math:`\text{high}`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    low = float(low) if low is not None else None
    high = float(high) if high is not None else None
    if context.executing_eagerly():
        return OpLib.execute('Clip', inputs, low=low, high=high)
    return OpLib.add('Clip', inputs, low=low, high=high, **kwargs)


@OpSchema.num_inputs(1)
def cos(inputs, **kwargs):
    r"""Compute the cos of input.

    .. math:: \text{out} = \cos(\text{input})

    Examples:

    ```python
    x = dragon.constant([0., math.pi])
    print(dragon.math.cos(x))  # [1., -1.]
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
        return OpLib.execute('Cos', inputs)
    return OpLib.add('Cos', inputs, **kwargs)


@OpSchema.num_inputs(1)
def cumsum(inputs, axis=0, exclusive=False, reverse=False, **kwargs):
    """Compute the cumulative sum of elements along the given axis.

    :attr:`axis` could be negative:

    ```python
    # A negative axis is the last-k axis
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])
    print(dragon.math.cumsum(x, axis=1))   # [[1, 3, 6], [4, 9, 15]]
    print(dragon.math.cumsum(x, axis=-1))  # Equivalent
    ```

    Use :attr:`exclusive` to exclude the top element:

    ```python
    x = dragon.constant([1, 2, 3])
    print(dragon.math.cumsum(x, exclusive=True))  # [0, 1, 3]
    ```

    Use :attr:`reverse` to reverse the cumulative direction:

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
        The axis to cumulate.
    exclusive : bool, optional, default=False
        ``True`` to exclude the top element.
    reverse : bool, optional, default=False
        ``True`` to cumulate in a reverse direction.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('CumSum', inputs, axis=axis,
                             exclusive=exclusive, reverse=reverse)
    return OpLib.add('CumSum', inputs, axis=axis,
                     exclusive=exclusive, reverse=reverse, **kwargs)


@OpSchema.num_inputs(2)
def div(inputs, **kwargs):
    r"""Compute the element-wise division.

    .. math:: \text{out} = \text{input1} \div \text{input2}

    Examples:

    ```python
    a = dragon.constant(4)
    b = dragon.constant(2)
    print(dragon.math.div([a, b]))
    print(a / b)  # Equivalent operation
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Div', inputs)
    return OpLib.add('Div', inputs, **kwargs)


@OpSchema.num_inputs(2)
def equal(inputs, **kwargs):
    r"""Compute the element-wise equal comparison.

    .. math:: \text{out} = (\text{input1} == \text{input2})

    Examples:

    ```python
    a = dragon.constant(2)
    b = dragon.constant(3)
    c = dragon.constant(3)
    print(dragon.math.equal([a, b]))
    print(dragon.math.equal([b, c]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Equal', inputs)
    return OpLib.add('Equal', inputs, **kwargs)


@OpSchema.num_inputs(1)
def exp(inputs, **kwargs):
    r"""Compute the exponential of input.

    .. math:: \text{out} = \exp(\text{input})

    Examples:

    ```python
    x = dragon.constant([1., 2., 3.])
    print(dragon.math.exp(x))  # [2.71828183, 7.3890561, 20.08553692]
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
        return OpLib.execute('Exp', inputs)
    return OpLib.add('Exp', inputs, **kwargs)


@OpSchema.num_inputs(1)
def floor(inputs, **kwargs):
    r"""Compute the largest integer not greater than input.

    .. math:: \text{out} = \lfloor \text{input} \rfloor

    Examples:

    ```python
    x = dragon.constant([0.9, 1.4, 1.9])
    print(dragon.math.floor(x))  # [0., 1., 1.]
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
        return OpLib.execute('Floor', inputs)
    return OpLib.add('Floor', inputs, **kwargs)


@OpSchema.num_inputs(2, 3)
def gemm(inputs, alpha=1, beta=1, transpose_a=False, transpose_b=False, **kwargs):
    r"""Compute the general matrix multiplication.

    .. math:: \text{out} = \alpha AB + \beta C

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The matrix :math:`A`, :math:`B` and optional :math:`C`.
    alpha : float, optional, default=1
        The value to :math:`\alpha`.
    beta : float, optional, default=1
        The value to :math:`\beta`.
    transpose_a : bool, optional, default=False
        ``True`` to transpose :math:`A` before computation.
    transpose_b : bool, optional, default=False
        ``True`` to transpose :math:`B` before computation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    alpha, beta = float(alpha), float(beta)
    if context.executing_eagerly():
        return OpLib.execute(
            'Gemm', inputs, alpha=alpha, beta=beta,
            transA=transpose_a, transB=transpose_b)
    return OpLib.add('Gemm', inputs, alpha=alpha, beta=beta,
                     transA=transpose_a, transB=transpose_b, **kwargs)


@OpSchema.num_inputs(2)
def greater(inputs, **kwargs):
    r"""Compute the element-wise greater comparison.

    .. math:: \text{out} = (\text{input1} > \text{input2})

    Examples:

    ```python
    a = dragon.constant(2)
    b = dragon.constant(3)
    c = dragon.constant(3)
    print(dragon.math.greater([a, b]))
    print(dragon.math.greater([b, c]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Greater', inputs)
    return OpLib.add('Greater', inputs, **kwargs)


@OpSchema.num_inputs(2)
def greater_equal(inputs, **kwargs):
    r"""Compute the element-wise greater-equal comparison.

    .. math:: \text{out} = (\text{input1} >= \text{input2})

    Examples:

    ```python
    a = dragon.constant(2)
    b = dragon.constant(3)
    c = dragon.constant(3)
    print(dragon.math.greater_equal([a, b]))
    print(dragon.math.greater_equal([b, c]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('GreaterEqual', inputs)
    return OpLib.add('GreaterEqual', inputs, **kwargs)


@OpSchema.num_inputs(1)
def invert(inputs, **kwargs):
    r"""Invert each bit of input.

    .. math:: \text{out} = \,\,\sim \text{input}

    Examples:

    ```python
    # Typically, ``x`` is a bool tensor
    print(dragon.bitwise.invert(dragon.constant([0, 1], 'bool')))  # [True, False]

    # Otherwise, integral types are required (unsigned or signed)
    # 00001101 (13) -> 11110010 (?)
    print(dragon.bitwise.invert(dragon.constant(13, 'uint8')))  # 242
    print(dragon.bitwise.invert(dragon.constant(13, 'int8')))   # -14
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
        return OpLib.execute('BitwiseNot', inputs)
    return OpLib.add('BitwiseNot', inputs, **kwargs)


@OpSchema.num_inputs(1)
def is_finite(inputs, **kwargs):
    r"""Check if the elements of input are finite.

    .. math:: \text{out} = \text{isfinite}(\text{input})

    Examples:

    ```python
    x = dragon.constant([0., float('nan'), float('inf')])
    print(dragon.math.is_finite(x))  # [True, False, False]
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
        return OpLib.execute('IsFinite', inputs)
    return OpLib.add('IsFinite', inputs, **kwargs)


@OpSchema.num_inputs(1)
def is_inf(inputs, **kwargs):
    r"""Check if the elements of input are infinite.

    .. math:: \text{out} = \text{isinf}(\text{input})

    Examples:

    ```python
    x = dragon.constant([0., 1., float('inf')])
    print(dragon.math.is_inf(x))  # [False, False, True]
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
        return OpLib.execute('IsInf', inputs)
    return OpLib.add('IsInf', inputs, **kwargs)


@OpSchema.num_inputs(1)
def is_nan(inputs, **kwargs):
    r"""Check if the elements of input are NaN.

    .. math:: \text{out} = \text{isnan}(\text{input})

    Examples:

    ```python
    x = dragon.constant([0., 1., float('nan')])
    print(dragon.math.is_nan(x))  # [False, False, True]
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
        return OpLib.execute('IsNaN', inputs)
    return OpLib.add('IsNaN', inputs, **kwargs)


@OpSchema.num_inputs(2)
def less(inputs, **kwargs):
    r"""Compute the element-wise less comparison.

    .. math:: \text{out} = (\text{input1} < \text{input2})

    Examples:

    ```python
    a = dragon.constant(2)
    b = dragon.constant(3)
    c = dragon.constant(3)
    print(dragon.math.less([a, b]))
    print(dragon.math.less([b, c]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Less', inputs)
    return OpLib.add('Less', inputs, **kwargs)


@OpSchema.num_inputs(2)
def less_equal(inputs, **kwargs):
    r"""Compute the element-wise less-equal comparison.

    .. math:: \text{out} = (\text{input1} <= \text{input2})

    Examples:

    ```python
    a = dragon.constant(2)
    b = dragon.constant(3)
    c = dragon.constant(3)
    print(dragon.math.less_equal([a, b]))
    print(dragon.math.less_equal([b, c]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('LessEqual', inputs)
    return OpLib.add('LessEqual', inputs, **kwargs)


@OpSchema.num_inputs(1)
def log(inputs, **kwargs):
    r"""Compute the logarithm of input.

    .. math:: \text{out} = \log(\text{input})

    Examples:

    ```python
    x = dragon.constant([1., 2., 3.])
    print(dragon.math.log(x))  # [0., 0.69314718, 1.09861229]
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
        return OpLib.execute('Log', inputs)
    return OpLib.add('Log', inputs, **kwargs)


@OpSchema.num_inputs(2)
def logical_and(inputs, **kwargs):
    r"""Compute the element-wise AND logical operation.

    .. math:: \text{out} = \text{input1} \mathbin{\&} \text{input2}

    Examples:

    ```python
    a = dragon.constant([False, True, False, True])
    b = dragon.constant([False, True, True, False])
    c = dragon.constant([0, 1, 0, 2])
    d = dragon.constant([0, 3, 4, 0])
    print(dragon.math.logical_and([a, b]))  # [False, True, False, False]
    print(dragon.math.logical_and([c, d]))  # [False, True, False, False]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('And', inputs)
    return OpLib.add('And', inputs, **kwargs)


@OpSchema.num_inputs(1)
def logical_not(inputs, **kwargs):
    r"""Compute the element-wise NOT logical operation.

    .. math:: \text{out} = \,\,\sim \text{input}

    Examples:

    ```python
    a = dragon.constant([False, True, True])
    b = dragon.constant([0, 1, 2])
    print(dragon.math.logical_not(a))  # [True, False, False]
    print(dragon.math.logical_not(b))  # [True, False, False]
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
        return OpLib.execute('Not', inputs)
    return OpLib.add('Not', inputs, **kwargs)


@OpSchema.num_inputs(2)
def logical_or(inputs, **kwargs):
    r"""Compute the element-wise OR logical operation.

    .. math:: \text{out} = \text{input1} \mathbin{|} \text{input2}

    Examples:

    ```python
    a = dragon.constant([False, True, False, True])
    b = dragon.constant([False, True, True, False])
    c = dragon.constant([0, 1, 0, 2])
    d = dragon.constant([0, 3, 4, 0])
    print(dragon.math.logical_or([a, b]))  # [False, True, True, True]
    print(dragon.math.logical_or([c, d]))  # [False, True, True, True]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Or', inputs)
    return OpLib.add('Or', inputs, **kwargs)


@OpSchema.num_inputs(2)
def logical_xor(inputs, **kwargs):
    r"""Compute the element-wise XOR logical operation.

    .. math:: \text{out} = \text{input1} \oplus \text{input2}

    Examples:

    ```python
    a = dragon.constant([False, True, False, True])
    b = dragon.constant([False, True, True, False])
    c = dragon.constant([0, 1, 0, 2])
    d = dragon.constant([0, 3, 4, 0])
    print(dragon.math.logical_xor([a, b]))  # [False, False, True, True]
    print(dragon.math.logical_xor([c, d]))  # [False, False, True, True]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Xor', inputs)
    return OpLib.add('Xor', inputs, **kwargs)


@OpSchema.num_inputs(2)
def matmul(inputs, **kwargs):
    r"""Compute the matrix multiplication.

    .. math:: \text{out} = \text{input1} \times \text{input2}

    The behavior depends on the shape of input tensors:

    * If both tensors are 1d, computes the vector product.
    * If tensors are 1d and >=2d, computes the vector-matrix multiplication.
    * If tensors are >=2d and 1d, computes the matrix-vector multiplication.
    * If both tensors are >= 2d, computes the matrix-matrix multiplication.
    * If one tensor is >= 3d, applies batching and broadcasting to the computation.

    Examples:

    ```python
    # Vector x Vector
    a = dragon.ones((2,), 'float32')
    b = dragon.ones((2,), 'float32')
    print(dragon.math.matmul([a, b]))
    # Vector x Matrix
    a = dragon.ones((2,), 'float32')
    b = dragon.ones((2, 3), 'float32')
    print(dragon.math.matmul([a, b]))
    # Matrix x Vector
    a = dragon.ones((3, 2), 'float32')
    b = dragon.ones((2,), 'float32')
    print(dragon.math.matmul([a, b]))
    # Matrix x Matrix
    a = dragon.ones((2, 3), 'float32')
    b = dragon.ones((3, 2), 'float32')
    print(dragon.math.matmul([a, b]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input tensors.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('MatMul', inputs)
    return OpLib.add('MatMul', inputs, **kwargs)


@OpSchema.num_inputs(1)
def max(inputs, axis=None, keepdims=False, **kwargs):
    """Compute the max value of elements along the given axis.

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.math.max(x, 1))
    print(dragon.math.max(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.max(x))  # 6

    # Also, ``axis`` could be a sequence of integers
    print(dragon.math.max(x, (0, 1)))  # 6
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    axes = None if axis is None else nest.flatten(axis)
    if context.executing_eagerly():
        return OpLib.execute('ReduceMax', inputs, axes=axes, keepdims=keepdims)
    return OpLib.add('ReduceMax', inputs,
                     axes=axes, keepdims=keepdims, **kwargs)


@OpSchema.num_inputs(2)
def maximum(inputs, **kwargs):
    r"""Compute the maximum value of given two inputs.

    .. math:: \text{out} = \max(\text{input1}, \text{input2})

    Parameters
    ----------
    inputs : Sequence[Union[dragon.Tensor, number]]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Maximum', inputs)
    return OpLib.add('Maximum', inputs, **kwargs)


@OpSchema.num_inputs(1)
def mean(inputs, axis=None, keepdims=False, **kwargs):
    """Compute the mean value of elements along the given axis.

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]], dtype='float32')

    # A negative axis is the last-k axis
    print(dragon.math.mean(x, 1))
    print(dragon.math.mean(x, -1))  # Equivalent

    # If axis is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.mean(x))  # 3.5

    # Also, axis could be a sequence of integers
    print(dragon.math.mean(x, (0, 1)))  # 3.5
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    axes = None if axis is None else nest.flatten(axis)
    if context.executing_eagerly():
        return OpLib.execute('ReduceMean', inputs, axes=axes, keepdims=keepdims)
    return OpLib.add('ReduceMean', inputs,
                     axes=axes, keepdims=keepdims, **kwargs)


@OpSchema.num_inputs(1)
def min(inputs, axis=None, keepdims=False, **kwargs):
    """Compute the min value of elements along the given axis.

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.math.min(x, 1))
    print(dragon.math.min(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.min(x))  # 1

    # Also, ``axis`` could be a sequence of integers
    print(dragon.math.min(x, (0, 1)))  # 1
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    axes = None if axis is None else nest.flatten(axis)
    if context.executing_eagerly():
        return OpLib.execute('ReduceMin', inputs, axes=axes, keepdims=keepdims)
    return OpLib.add('ReduceMin', inputs,
                     axes=axes, keepdims=keepdims, **kwargs)


@OpSchema.num_inputs(2)
def minimum(inputs, **kwargs):
    r"""Compute the minimum value of given two inputs.

    .. math:: \text{out} = \min(\text{input1}, \text{input2})

    Parameters
    ----------
    inputs : Sequence[Union[dragon.Tensor, number]]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Minimum', inputs)
    return OpLib.add('Minimum', inputs, **kwargs)


@OpSchema.num_inputs(1)
def moments(inputs, axis=None, keepdims=False, **kwargs):
    r"""Compute the mean and variance of input along the given axis.

    .. math::
        \begin{cases}
            \mathrm{E}[x] = \frac{1}{n}\sum(x) \\
            \mathrm{Var}[x] = \frac{1}{n}\sum(x - \mathrm{E}[x])^{2}
        \end{cases}

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]], dtype='float32')

    # A negative axis is the last-k axis
    print(dragon.nn.moments(x, 1))
    print(dragon.nn.moments(x, -1))  # Equivalent

    # If axis is None, reduce as a vector and return scalars
    # will be applied to return a scalar result
    print(dragon.nn.moments(x))  # mean is 3.5, var is 2.916667

    # Also, axis could be a sequence of integers
    print(dragon.nn.moments(x, (0, 1)))  # mean is 3.5, var is 2.916667
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The mean tensor.
    dragon.Tensor
        The variance tensor.

    """
    axes = None if axis is None else nest.flatten(axis)
    if context.executing_eagerly():
        return OpLib.execute(
            'Moments', inputs, outputs=[None, None],
            axes=axes, keepdims=keepdims)
    return OpLib.add('Moments', inputs, num_outputs=2,
                     axes=axes, keepdims=keepdims, **kwargs)


@OpSchema.num_inputs(2)
def mul(inputs, **kwargs):
    r"""Compute the element-wise multiplication.

    .. math:: \text{out} = \text{input1} \times \text{input2}

    Examples:

    ```python
    a = dragon.constant(4)
    b = dragon.constant(2)
    print(dragon.math.mul([a, b]))
    print(a * b)  # Equivalent operation
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Mul', inputs)
    return OpLib.add('Mul', inputs, **kwargs)


@OpSchema.num_inputs(1)
def negative(inputs, **kwargs):
    r"""Compute the element-wise negative.

    .. math:: \text{out} = -\text{input}

    ```python
    x = dragon.constant([-1, 0, 1])
    print(dragon.math.negative(x))
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
        return OpLib.execute('Neg', inputs)
    return OpLib.add('Neg', inputs, **kwargs)


@OpSchema.num_inputs(1)
def norm(inputs, ord=None, axis=None, keepdims=False, **kwargs):
    """Compute the norm value of elements along the given axis.

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.constant([[1., 2., 3.], [4., 5., 6.]])

    # A negative axis is the last-k axis
    print(dragon.math.norm(x, axis=1))
    print(dragon.math.norm(x, axis=-1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.norm(x))  # 9.539

    # Also, ``axis`` could be a sequence of integers
    print(dragon.math.norm(x, axis=(0, 1)))  # 9.539
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    ord : {1, 2, 'fro'}, optional
        The norm order.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if ord is None or ord == 2 or ord == 'fro':
        op_type = 'ReduceL2'
    elif ord == 1:
        op_type = 'ReduceL1'
    else:
        raise ValueError('Unsupported norm order: ' + str(ord))
    axes = None if axis is None else nest.flatten(axis)
    if context.executing_eagerly():
        return OpLib.execute(op_type, inputs, axes=axes, keepdims=keepdims)
    return OpLib.add(op_type, inputs, axes=axes, keepdims=keepdims, **kwargs)


@OpSchema.num_inputs(2)
def not_equal(inputs, **kwargs):
    r"""Compute the element-wise not-equal comparison.

    .. math:: \text{out} = (\text{input1} \neq \text{input2})

    Examples:

    ```python
    a = dragon.constant(2)
    b = dragon.constant(3)
    c = dragon.constant(3)
    print(dragon.math.not_equal([a, b]))
    print(dragon.math.not_equal([b, c]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('NotEqual', inputs)
    return OpLib.add('NotEqual', inputs, **kwargs)


@OpSchema.num_inputs(2)
def pow(inputs, **kwargs):
    r"""Compute the power of input.

    .. math:: \text{out} = \text{input}^{\text{exponent}}

    The two inputs should be broadcast to each other:

    ```python
    x = dragon.fill(shape=(1, 2), value=2)
    print(dragon.math.pow([x, x]))  # [[4, 4]]
    print(dragon.math.pow([x, 3]))  # [[8, 8]]
    print(dragon.math.pow([3, x]))  # [[9, 9]]
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input and exponent tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Pow', inputs)
    return OpLib.add('Pow', inputs, **kwargs)


@OpSchema.num_inputs(1)
def reciprocal(inputs, **kwargs):
    r"""Compute the reciprocal of input.

    .. math:: \text{out} = \frac{1}{\text{input}}

    Examples:

    ```python
    x = dragon.constant([0., 1., 2.])
    print(dragon.math.reciprocal(x))  # [inf, 1., 0.5]
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
        return OpLib.execute('Reciprocal', inputs)
    return OpLib.add('Reciprocal', inputs, **kwargs)


@OpSchema.num_inputs(1)
def round(inputs, **kwargs):
    r"""Compute the nearest integer of input.

    .. math:: \text{out} = \lfloor \text{input} \rceil

    Examples:

    ```python
    x = dragon.constant([0.9, 1.4, 1.9])
    print(dragon.math.round(x))  # [1., 1., 2.]
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
        return OpLib.execute('Round', inputs)
    return OpLib.add('Round', inputs, **kwargs)


@OpSchema.num_inputs(1)
def rsqrt(inputs, **kwargs):
    r"""Compute the reciprocal square root of input.

    .. math:: \text{out} = \frac{1}{\sqrt{\text{input}}}

    Examples:

    ```python
    x = dragon.constant([0., 4., 16.])
    print(dragon.math.rsqrt(x))  # [inf, 0.5, 0.25]
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
        return OpLib.execute('Rsqrt', inputs)
    return OpLib.add('Rsqrt', inputs, **kwargs)


@OpSchema.num_inputs(1)
def sign(inputs, **kwargs):
    r"""Compute the sign indication of input.

    .. math::
        \text{out}_[i] =
            \begin{cases}
                -1, & \text{ if } \text{input}_{i} < 0 \\
                 0, & \text{ if } \text{input}_{i} = 0 \\
                 1, & \text{ if } \text{input}_{i} > 0
            \end{cases}

    Examples:

    ```python
    x = dragon.constant([-2, 0, 2])
    print(dragon.math.sign(x))  # [-1, 0, 1]
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
        return OpLib.execute('Sign', inputs)
    return OpLib.add('Sign', inputs, **kwargs)


@OpSchema.num_inputs(1)
def sin(inputs, **kwargs):
    r"""Compute the sin of input.

    .. math:: \text{out} = \sin(\text{input})

    Examples:

    ```python
    x = dragon.constant([0., math.pi / 2])
    print(dragon.math.sin(x))  # [0., 1.]
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
        return OpLib.execute('Sin', inputs)
    return OpLib.add('Sin', inputs, **kwargs)


@OpSchema.num_inputs(1)
def sqrt(inputs, **kwargs):
    r"""Compute the square root of input.

    .. math:: \text{out} = \sqrt{\text{input}}

    Examples:

    ```python
    x = dragon.constant([4., 9., 16.])
    print(dragon.math.sqrt(x))  # [2., 3., 4.]
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
        return OpLib.execute('Sqrt', inputs)
    return OpLib.add('Sqrt', inputs, **kwargs)


@OpSchema.num_inputs(1)
def square(inputs, **kwargs):
    r"""Compute the square of input.

    .. math:: \text{out} = \text{input}^{2}

    Examples:

    ```python
    x = dragon.constant([2, 3, 4])
    print(dragon.math.square(x))
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
        return OpLib.execute('Square', inputs)
    return OpLib.add('Square', inputs, **kwargs)


@OpSchema.num_inputs(2)
def sub(inputs, **kwargs):
    r"""Compute the element-wise subtraction.

    .. math:: \text{out} = \text{input1} - \text{input2}

    Examples:

    ```python
    a = dragon.constant(1)
    b = dragon.constant(2)
    print(dragon.math.sub([a, b]))
    print(a - b)  # Equivalent operation
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input1 and input2 tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute('Sub', inputs)
    return OpLib.add('Sub', inputs, **kwargs)


@OpSchema.num_inputs(1)
def sum(inputs, axis=None, keepdims=False, **kwargs):
    """Compute the sum value of elements along the given axis.

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(dragon.math.sum(x, 1))
    print(dragon.math.sum(x, -1))  # Equivalent

    # If axis is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.sum(x))  # 21

    # Also, axis could be a sequence of integers
    print(dragon.math.sum(x, (0, 1)))  # 21
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    axes = None if axis is None else nest.flatten(axis)
    if context.executing_eagerly():
        return OpLib.execute('ReduceSum', inputs, axes=axes, keepdims=keepdims)
    return OpLib.add('ReduceSum', inputs,
                     axes=axes, keepdims=keepdims, **kwargs)


@OpSchema.num_inputs(1)
def var(inputs, axis=None, keepdims=False, **kwargs):
    """Compute the variance value of elements along the given axis.

    :attr:`axis` could be negative or ``None``:

    ```python
    x = dragon.constant([[1, 2, 3], [4, 5, 6]], dtype='float32')

    # A negative axis is the last-k axis
    print(dragon.math.var(x, 1))
    print(dragon.math.var(x, -1))  # Equivalent

    # If axis is None, the vector-style reduction
    # will be applied to return a scalar result
    print(dragon.math.var(x))  # 3.5

    # Also, axis could be a sequence of integers
    print(dragon.math.var(x, (0, 1)))  # 3.5
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    axes = None if axis is None else nest.flatten(axis)
    if context.executing_eagerly():
        return OpLib.execute('ReduceVar', inputs, axes=axes, keepdims=keepdims)
    return OpLib.add('ReduceVar', inputs,
                     axes=axes, keepdims=keepdims, **kwargs)
