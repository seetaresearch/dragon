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

from dragon.core.eager import context
from dragon.core.framework import ops
from dragon.core.ops import math_ops_lib
from dragon.core.ops.utils import OpSchema
from dragon.core.ops.utils import parse_args


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Abs').apply([inputs])
    else:
        return op_lib.blend('Abs', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Add').apply(inputs)
    else:
        return op_lib.blend('Add', **args)


@OpSchema.num_inputs(2, 3)
def affine(inputs, axis=1, num_axes=1, **kwargs):
    r"""Compute the affine transformation along the given axes.

    .. math:: y = Wx + b

    The range of axes is defined as:

    .. math:: [\text{Axis}, \text{Axis} + \text{NumAxes})

    Set ``axis`` to specific the start axis.

    Set ``num_axes`` to -1 will scale all remained axes.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor **x**, **W** and **b**.
    axis : int, optional, default=1
        The start axis, can be negative.
    num_axes : int, optional, default=1
        The number of axes to compute.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = math_ops_lib.Affine
    if context.executing_eagerly():
        return op_lib \
            .instantiate(axis=axis, num_axes=num_axes) \
            .apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def axpby(inputs, outputs=None, alpha=1., beta=1., **kwargs):
    r"""Compute the element-wise addition from input to output.

    .. math:: \text{out} = \alpha * \text{input} + \beta * \text{out}

    If ``outputs`` is not provided, **zeros** will be used instead.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    outputs : dragon.Tensor, optional
        The output tensor.
    alpha : number, optional, default=1.
        The value to :math:`\alpha`.
    beta : number, optional, default=1.
        The value to :math:`\beta`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['alpha'], args['beta'] = float(alpha), float(beta)
    op_lib = math_ops_lib.Axpby
    if context.executing_eagerly():
        return op_lib \
            .instantiate(alpha=args['alpha'], beta=args['beta']) \
            .apply([inputs], [outputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2)
def bitwise_and(inputs, **kwargs):
    r"""Compute the element-wise AND bitwise operation.

    .. math:: \text{out} = \text{input1} \mathbin{\&} \text{input2}

    Examples:

    ```python
    a = dragon.constant([False, True, False, True])
    b = dragon.constant([False, True, True, False])
    print(dragon.bitwise.bitwise_and([a, b]))  # False, True, False, False
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
    return mul(inputs, **kwargs)


@OpSchema.num_inputs(2)
def bitwise_or(inputs, **kwargs):
    r"""Compute the element-wise OR bitwise operation.

    .. math:: \text{out} = \text{input1} \mathbin{|} \text{input2}

    Examples:

    ```python
    a = dragon.constant([False, True, False, True])
    b = dragon.constant([False, True, True, False])
    print(dragon.bitwise.bitwise_or([a, b]))  # False, True, True, True
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
    return add(inputs, **kwargs)


@OpSchema.num_inputs(2)
def bitwise_xor(inputs, **kwargs):
    r"""Compute the element-wise XOR bitwise operation.

    .. math:: \text{out} = \text{input1} \oplus \text{input2}

    Examples:

    ```python
    a = dragon.constant([False, True, False, True])
    b = dragon.constant([False, True, True, False])
    print(dragon.bitwise.bitwise_xor([a, b]))  # False, False, True, True
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
    return sub(inputs, **kwargs)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Ceil').apply([inputs])
    else:
        return op_lib.blend('Ceil', **args)


@OpSchema.num_inputs(1)
def clip(inputs, low=None, high=None, **kwargs):
    r"""Compute the clipped input according to the given bounds.

    .. math:: \text{out} = \min(\max(\text{input}, \text{low}), \text{high})

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
    args = parse_args(locals())
    if low is not None:
        args['low'] = float(args['low'])
    if high is not None:
        args['high'] = float(args['high'])
    op_lib = math_ops_lib.Clip
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                low=args['low'],
                high=args['high'],
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Cos').apply([inputs])
    else:
        return op_lib.blend('Cos', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Div').apply(inputs)
    else:
        return op_lib.blend('Div', **args)


@OpSchema.num_inputs(2)
def dot(inputs, **kwargs):
    r"""Compute the dot product.

    .. math:: \text{out} = \text{input1} \cdot \text{input2}

    If ``rank(input1)`` == ``rank(input2)`` == 1, compute vector product:

    ```python
    a = dragon.ones((2,))
    b = dragon.ones((2,))
    print(dragon.math.dot([a, b]))  # 2.0
    ```

    If ``rank(input1)`` == ``rank(input2)`` == 2, compute matrix multiplication:

    ```python
    a = dragon.ones((2, 3))
    b = dragon.ones((3, 2))
    print(dragon.math.dot([a, b]))  # [[[3. 3.], [3. 3.]]]
    print(dragon.math.matmul([a, b]))  # Equivalent
    ```

    If ``rank(input1)`` >= 2, ``rank(input2)`` == 1, compute matrix-vector multiplication:

    ```python
    a = dragon.ones((2, 3))
    b = dragon.ones((3,))
    print(dragon.math.dot([a, b]))  # [[3. 3.]]
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
    args = parse_args(locals())
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Dot').apply(inputs)
    else:
        return op_lib.blend('Dot', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Equal').apply(inputs)
    else:
        return op_lib.blend('Equal', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Exp').apply([inputs])
    else:
        return op_lib.blend('Exp', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Floor').apply([inputs])
    else:
        return op_lib.blend('Floor', **args)


@OpSchema.num_inputs(2, 3)
def fully_connected(inputs, axis=1, transpose_w=True, **kwargs):
    r"""Compute the dense matrix multiplication along the given axes.

    .. math:: y = Wx + b

    The column of input matrix is determined by:

    .. math:: \text{Col} = \text{DimSince}(\text{Input}, \text{Axis})

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor :math:`x`, :math:`W` and :math:`b`.
    axis : int, optional, default=1
        The start axis to compute, can be negative.
    transpose_w : bool, optional, default=True
        **True** to transpose :math:`W` before computation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = math_ops_lib.FullyConnected
    if context.executing_eagerly():
        return op_lib \
            .instantiate(axis=axis, transpose_w=transpose_w) \
            .apply(inputs)
    else:
        args.pop('transpose_w')
        args['transW'] = transpose_w
        return op_lib.blend('FullyConnected', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.BinaryOp
    inputs = ops.remove_binary_scalar(inputs)
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Greater').apply(inputs)
    else:
        return op_lib.blend('Greater', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='GreaterEqual').apply(inputs)
    else:
        return op_lib.blend('GreaterEqual', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Invert').apply([inputs])
    else:
        return op_lib.blend('Invert', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='IsInf').apply([inputs])
    else:
        return op_lib.blend('IsInf', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='IsNaN').apply([inputs])
    else:
        return op_lib.blend('IsNaN', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Log').apply([inputs])
    else:
        return op_lib.blend('Log', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Less').apply(inputs)
    else:
        return op_lib.blend('Less', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='LessEqual').apply(inputs)
    else:
        return op_lib.blend('LessEqual', **args)


@OpSchema.num_inputs(2)
def matmul(inputs, transpose_a=False, transpose_b=False, **kwargs):
    r"""Compute the matrix multiplication.

    .. math:: y = a \times b

    The rank of ``a`` and ``b`` should be equal and >= 2:

    ```python
    # Ok, a typical matrix multiplication
    a = dragon.ones((2, 3), 'float32')
    b = dragon.ones((3, 3), 'float32')
    print(dragon.math.matmul([a, b]))

    # Compute a batch matrix multiplication if rank > 2
    aa = dragon.ones((4, 2, 3), 'float32')
    bb = dragon.ones((4, 3, 3), 'float32')
    print(dragon.math.matmul([aa, bb]))
    ```

    If inputs are transposed, remember to transpose them back:

    ```python
    a = dragon.ones((3, 2), 'float32')
    b = dragon.ones((3, 3), 'float32')
    print(dragon.math.matmul([a, b]))  # ``a`` takes the wrong dimensions
    print(dragon.math.matmul([a, b], transpose_a=True))  # Ok
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The matrix :math:`a` and :math:`b`.
    transpose_a : bool, optional, default=False
        **True** to transpose :math:`a` before computation.
    transpose_b : bool, optional, default=False
        **True** to transpose :math:`b` before computation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    op_lib = math_ops_lib.MatMul
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                transpose_a=transpose_a,
                transpose_b=transpose_b,
            ).apply(inputs)
    else:
        args.pop('transpose_a')
        args.pop('transpose_b')
        return op_lib.blend(transA=transpose_a, transB=transpose_b, **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Maximum').apply(inputs)
    else:
        return op_lib.blend('Maximum', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Minimum').apply(inputs)
    else:
        return op_lib.blend('Minimum', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Mul').apply(inputs)
    else:
        return op_lib.blend('Mul', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Neg').apply([inputs])
    else:
        return op_lib.blend('Neg', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='NotEqual').apply(inputs)
    else:
        return op_lib.blend('NotEqual', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Pow').apply(inputs)
    else:
        return op_lib.blend('Pow', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Reciprocal').apply([inputs])
    else:
        return op_lib.blend('Reciprocal', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Round').apply([inputs])
    else:
        return op_lib.blend('Round', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Rsqrt').apply([inputs])
    else:
        return op_lib.blend('Rsqrt', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Sign').apply([inputs])
    else:
        return op_lib.blend('Sign', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Sin').apply([inputs])
    else:
        return op_lib.blend('Sin', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Sqrt').apply([inputs])
    else:
        return op_lib.blend('Sqrt', **args)


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
    args = parse_args(locals())
    op_lib = math_ops_lib.UnaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Square').apply([inputs])
    else:
        return op_lib.blend('Square', **args)


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
    args = parse_args(locals())
    inputs = ops.remove_binary_scalar(inputs)
    op_lib = math_ops_lib.BinaryOp
    if context.executing_eagerly():
        return op_lib.instantiate(op_type='Sub').apply(inputs)
    else:
        return op_lib.blend('Sub', **args)
