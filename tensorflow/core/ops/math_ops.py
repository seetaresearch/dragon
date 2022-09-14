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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_ops.py>
#
# ------------------------------------------------------------
"""Math ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import activation_ops
from dragon.core.ops import constant_ops
from dragon.core.ops import math_ops


def abs(x, name=None):
    r"""Compute the absolute value of input.

    .. math:: \text{out} = \left| \text{input} \right|

    Examples:

    ```python
    print(tf.math.abs(tf.constant([-1, 0, 1])))  # [1, 0, 1]
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

    """
    return math_ops.abs(x, name=name)


def add(x, y, name=None):
    r"""Compute the element-wise addition.

    .. math:: \text{out} = \text{input1} + \text{input2}

    ```python
    x = tf.constant(1)
    y = tf.constant(2)
    print(tf.math.add(x, y))
    print(x + y)  # Equivalent operation
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.add([x, y], name=name)


def add_n(inputs, name=None):
    r"""Compute the element-wise sum on a sequence of inputs.

    .. math:: \text{out} = \sum(\text{input}_{i})

    Examples:

    ```python
    x = tf.constant([1, 2, 3])
    print(tf.math.add_n([x, x, x]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input tensors.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    output = inputs[0]
    for input in inputs[1:]:
        if output.id == input.id:
            output = output + input
        else:
            output += input
    return output


def argmax(input, axis=None, name=None, **kwargs):
    """Compute the index of maximum elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]])

    # A negative ``axis`` is the last-k axis
    print(tf.math.argmax(x, 1))
    print(tf.math.argmax(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar index
    print(tf.math.argmax(x))
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to reduce.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The index of maximum elements.

    """
    kwargs.pop('output_type', None)
    return math_ops.argmax(input, axis=axis, name=name)


def argmin(input, axis=None, name=None):
    """Compute the index of minimum elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]])

    # A negative ``axis`` is the last-k axis
    print(tf.math.argmin(x, 1))
    print(tf.math.argmin(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar index
    print(tf.math.argmin(x))
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to reduce.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The index of minimum elements.

    """
    return math_ops.argmin(input, axis=axis, name=name)


def atan2(y, x, name=None):
    r"""Compute the element-wise arc-tangent of two arguments.

    .. math:: \text{out} = \text{arctan}(\frac{\text{input1}}{\text{input2}})

    ```python
    y = tf.constant(1.)
    x = tf.constant(2.)
    print(tf.math.atan2(y, x))  # 0.46364761
    ```

    Parameters
    ----------
    y : dragon.Tensor
        The input1 tensor.
    x : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.atan2([y, x], name=name)


def cast(x, dtype, name=None):
    """Cast the data type of input.

    Examples:

    ```python
    x = tf.constant([1, 2, 3], dtype='int64')
    print(tf.cast(x, dtype='float32'))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    dtype : str
        The data type to cast to.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dtype = str(dtype) if dtype else dtype
    return math_ops.cast(x, dtype=dtype, name=name)


def ceil(x, name=None):
    r"""Compute the smallest integer not less than input.

    .. math:: \text{out} = \lceil \text{input} \rceil

    Examples:

    ```python
    x = tf.constant([1.4, 1.7, 2.0])
    print(tf.math.ceil(x))  # [2., 2., 2.]
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

    """
    return math_ops.ceil(x, name=name)


def cos(x, name=None):
    r"""Compute the cos of input.

    .. math:: \text{out} = \cos(\text{input})

    Examples:

    ```python
    x = tf.constant([0., math.pi])
    print(tf.math.cos(x))  # [1., -1.]
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

    """
    return math_ops.cos(x, name=name)


def cumsum(x, axis=0, exclusive=False, reverse=False, name=None):
    """Compute the cumulative sum of elements along the given axis.

    The argument ``axis`` could be negative:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(tf.math.cumsum(x, 1))   # [[1, 3, 6], [4, 9, 15]]
    print(tf.math.cumsum(x, -1))  # Equivalent
    ```

    To exclude the top element, set the ``exclusive``:

    ```python
    x = tf.constant([1, 2, 3])
    print(tf.math.cumsum(x, exclusive=True))  # [0, 1, 3]
    ```

    Also, ``reverse`` could be set to reverse the cumulative direction:

    ```python
    x = tf.constant([1, 2, 3])
    print(tf.math.cumsum(x))  # [1, 3, 6]
    print(tf.math.cumsum(x, reverse=True))  # [6, 5, 3]
    print(tf.math.cumsum(x, exclusive=True, reverse=True))  # [5, 3, 0]
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    axis : int, optional, default=0
        The cumulative axis.
    exclusive : bool, optional, default=False
        ``True`` to exclude the top element.
    reverse : bool, optional, default=False
        ``True`` to compute in the reverse direction.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.cumsum(
        x, axis, exclusive=exclusive, reverse=reverse, name=name)


def divide(x, y, name=None):
    r"""Compute the element-wise division.

    .. math:: \text{out} = \text{input1} \div \text{input2}

    Examples:

    ```python
    x = tf.constant(4)
    y = tf.constant(2)
    print(tf.math.divide(x, y))
    print(x / y)  # Equivalent operation
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.div([x, y], name=name)


def equal(x, y, name=None):
    r"""Compute the element-wise equal comparison.

    .. math:: \text{out} = (\text{input1} == \text{input2})

    Examples:

    ```python
    x = tf.constant(2)
    y = tf.constant(3)
    z = tf.constant(3)
    print(tf.math.equal(x, y))
    print(tf.math.equal(y, z))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.equal([x, y], name=name)


def exp(x, name=None):
    r"""Compute the exponential of input.

    .. math:: \text{out} = \exp(\text{input})

    Examples:

    ```python
    x = tf.constant([1., 2., 3.])
    print(tf.math.exp(x))  # [2.71828183, 7.3890561, 20.08553692]
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

    """
    return math_ops.exp(x, name=name)


def floor(x, name=None):
    r"""Compute the largest integer not greater than input.

    .. math:: \text{out} = \lfloor \text{input} \rfloor

    Examples:

    ```python
    x = tf.constant([0.9, 1.4, 1.9])
    print(tf.math.floor(x))  # [0., 1., 1.]
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

    """
    return math_ops.floor(x, name=name)


def greater(x, y, name=None):
    r"""Compute the element-wise greater comparison.

    .. math:: \text{out} = (\text{input1} > \text{input2})

    Examples:

    ```python
    x = tf.constant(2)
    y = tf.constant(3)
    z = tf.constant(3)
    print(tf.math.greater(x, y))
    print(tf.math.greater(y, z))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.greater([x, y], name=name)


def greater_equal(x, y, name=None):
    r"""Compute the element-wise greater-equal comparison.

    .. math:: \text{out} = (\text{input1} >= \text{input2})

    Examples:

    ```python
    x = tf.constant(2)
    y = tf.constant(3)
    z = tf.constant(3)
    print(tf.math.greater_equal(x, y))
    print(tf.math.greater_equal(y, z))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.greater_equal([x, y], name=name)


def is_finite(x, name=None):
    r"""Check if the elements of input are finite.

    .. math:: \text{out} = \text{isfinite}(\text{input})

    Examples:

    ```python
    x = tf.constant([0., float('nan'), float('inf')])
    print(tf.math.is_finite(x))  # [True, False, False]
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

    """
    return math_ops.is_finite(x, name=name)


def is_inf(x, name=None):
    r"""Check if the elements of input are infinite.

    .. math:: \text{out} = \text{isinf}(\text{input})

    Examples:

    ```python
    x = tf.constant([0., 1., float('inf')])
    print(tf.math.is_inf(x))  # [False, False, True]
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

    """
    return math_ops.is_inf(x, name=name)


def is_nan(x, name=None):
    r"""Check if the elements of input are NaN.

    .. math:: \text{out} = \text{isnan}(\text{input})

    Examples:

    ```python
    x = tf.constant([0., 1., float('nan')])
    print(tf.math.is_nan(x))  # [False, False, True]
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

    """
    return math_ops.is_nan(x, name=name)


def less(x, y, name=None):
    r"""Compute the element-wise less comparison.

    .. math:: \text{out} = (\text{input1} < \text{input2})

    Examples:

    ```python
    x = tf.constant(2)
    y = tf.constant(3)
    z = tf.constant(3)
    print(tf.math.less(x, y))
    print(tf.math.less(y, z))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.less([x, y], name=name)


def less_equal(x, y, name=None):
    r"""Compute the element-wise less-equal comparison.

    .. math:: \text{out} = (\text{input1} <= \text{input2})

    Examples:

    ```python
    x = tf.constant(2)
    y = tf.constant(3)
    z = tf.constant(3)
    print(tf.math.less_equal(x, y))
    print(tf.math.less_equal(y, z))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.less_equal([x, y], name=name)


def linspace(start, stop, num, dtype='int64', name=None, axis=0):
    r"""Generate evenly spaced values within intervals along the given axis.

    Interval :math:`[\text{start}, \text{stop})` is determined for ``num`` values:

    ```python
    x = tf.linspace(2, 4, num=3)  # [2, 3, 4]
    ```

    More intervals are accepted to generate N-d coordinates:

    ```python
    x = tf.linspace([1, 2], [3, 4], num=3, axis=0)  # [[1, 2], [2, 3], [3, 4]]
    y = tf.linspace([1, 2], [3, 4], num=3, axis=1)  # [[1, 2, 3], [2, 3, 4]]
    ```

    Parameters
    ----------
    start : Union[number, Sequence[number]]
        The start(s) of interval.
    stop: Union[number, Sequence[number]]
        The stop(s) of interval.
    num : int
        The number of values to generate.
    dtype : str, optional, default='int64'
        The optional data type.
    name : str, optional
        The operation name.
    axis : int, optional, default=0
        The axis to generate values.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dtype = str(dtype) if dtype else dtype
    return constant_ops.linspace(
        start, stop, num, dtype=dtype, name=name, axis=axis)


def log(x, name=None):
    r"""Compute the logarithm of input.

    .. math:: \text{out} = \log(\text{input})

    Examples:

    ```python
    x = tf.constant([1., 2., 3.])
    print(tf.math.log(x))  # [0., 0.69314718, 1.09861229]
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

    """
    return math_ops.log(x, name=name)


def matmul(a, b, name=None):
    r"""Compute the matrix multiplication.

    .. math:: \text{out} = a \times b

    The behavior depends on the shape of input tensors:

    * If both tensors are 1d, computes the vector product.
    * If tensors are 1d and >=2d, computes the vector-matrix multiplication.
    * If tensors are >=2d and 1d, computes the matrix-vector multiplication.
    * If both tensors are >= 2d, computes the matrix-matrix multiplication.
    * If one tensor is >= 3d, applies batching and broadcasting to the computation.

    Examples:

    ```python
    # Vector x Vector
    a = tf.ones((2,), 'float32')
    b = tf.ones((2,), 'float32')
    print(tf.linalg.matmul(a, b))
    # Vector x Matrix
    a = tf.ones((2,), 'float32')
    b = tf.ones((2, 3), 'float32')
    print(tf.linalg.matmul(a, b))
    # Matrix x Vector
    a = tf.ones((3, 2), 'float32')
    b = tf.ones((2,), 'float32')
    print(tf.linalg.matmul(a, b))
    # Matrix x Matrix
    a = tf.ones((2, 3), 'float32')
    b = tf.ones((3, 2), 'float32')
    print(tf.linalg.matmul(a, b))
    ```

    Parameters
    ----------
    a : dragon.Tensor
        The matrix :math:`a`.
    b : dragon.Tensor
        The matrix :math:`b`.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.matmul([a, b], name=name)


def multiply(x, y, name=None):
    r"""Compute the element-wise multiplication.

    .. math:: \text{out} = \text{input1} \times \text{input2}

    Examples:

    ```python
    x = tf.constant(4)
    y = tf.constant(2)
    print(tf.math.multiply(x, y))
    print(x * y)  # Equivalent operation
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.mul([x, y], name=name)


def negative(x, name=None):
    r"""Compute the element-wise negative.

    .. math:: \text{out} = -\text{input}

    ```python
    x = tf.constant([-1, 0, 1])
    print(tf.math.negative(x))
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

    """
    return math_ops.negative(x, name=name)


def not_equal(x, y, name=None):
    r"""Compute the element-wise not-equal comparison.

    .. math:: \text{out} = (\text{input1} != \text{input2})

    Examples:

    ```python
    x = tf.constant(2)
    y = tf.constant(3)
    z = tf.constant(3)
    print(tf.math.not_equal(x, y))
    print(tf.math.not_equal(y, z))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.not_equal([x, y], name=name)


def pow(x, y, name=None):
    r"""Compute the power of input.

    .. math:: \text{out} = \text{input}^{\text{exponent}}

    The two inputs should be broadcast to each other:

    ```python
    x = tf.fill(dims=(1, 2), value=2)
    print(tf.math.pow(x, x))  # [[4, 4]]
    print(tf.math.pow(x, 3))  # [[8, 8]]
    print(tf.math.pow(3, x))  # [[9, 9]]
    ```

    Parameters
    ----------
    x : Union[dragon.Tensor, number]
        The input tensor.
    y : Union[dragon.Tensor, number]
        The exponent tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.pow([x, y], name=name)


def range(start, limit=None, delta=1, dtype='int64', name=None):
    r"""Return a tensor of evenly spaced values within an interval.

    Specify ``start`` and ``limit`` to determine an interval:

    ```python
    x = tf.range(2, 4)  # [2, 3]
    ```

    If ``limit`` is **None**, interval :math:`[0, start)` will be taken instead:

    ```python
    x = tf.range(5)  # [0, 1, 2, 3, 4]
    ```

    Set ``delta`` to make the strides:

    ```python
    x = tf.range(5, delta=2)  # [0, 2, 4]
    ```

    Parameters
    ----------
    start : number
        The start of interval.
    limit : number, optional
        The end of interval.
    delta : number, optional, default=1
        The spacing between two elements.
    dtype : str, optional, default='int64'
        The optional data type.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dtype = str(dtype) if dtype else dtype
    return constant_ops.range(start, limit, delta, dtype=dtype, name=name)


def reciprocal(x, name=None):
    r"""Compute the reciprocal of input.

    .. math:: \text{out} = \frac{1}{\text{input}}

    Examples:

    ```python
    x = tf.constant([0., 1., 2.])
    print(tf.math.reciprocal(x))  # [inf, 1., 0.5]
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

    """
    return math_ops.reciprocal(x, name=name)


def reduce_max(input_tensor, axis=None, keepdims=False, name=None):
    """Compute the max value of elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(tf.math.reduce_max(x, 1))
    print(tf.math.reduce_max(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(tf.math.reduce_max(x))  # 6
    ```

    Parameters
    ----------
    input_tensor : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.max(input_tensor, axis, keepdims=keepdims, name=name)


def reduce_mean(input_tensor, axis=None, keepdims=False, name=None):
    """Compute the mean value of elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype='float32')

    # A negative axis is the last-k axis
    print(tf.math.reduce_mean(x, 1))
    print(tf.math.reduce_mean(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(tf.math.reduce_mean(x))  # 3.5

    # Also, ``axis`` could be a sequence of integers
    print(tf.math.reduce_mean(x, [0, 1]))  # 3.5
    ```

    Parameters
    ----------
    input_tensor : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.mean(input_tensor, axis, keepdims=keepdims, name=name)


def reduce_min(input_tensor, axis=None, keepdims=False, name=None):
    """Compute the min value of elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(tf.math.reduce_min(x, 1))
    print(tf.math.reduce_min(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(tf.math.reduce_min(x))  # 1
    ```

    Parameters
    ----------
    input_tensor : dragon.Tensor
        The input tensor.
    axis : int, optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.min(input_tensor, axis, keepdims=keepdims, name=name)


def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
    """Compute the sum value of elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(tf.math.reduce_sum(x, 1))
    print(tf.math.reduce_sum(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(tf.math.reduce_sum(x))  # 21

    # Also, ``axis`` could be a sequence of integers
    print(tf.math.reduce_sum(x, [0, 1]))  # 21
    ```

    Parameters
    ----------
    input_tensor : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.sum(input_tensor, axis, keepdims=keepdims, name=name)


def reduce_variance(input_tensor, axis=None, keepdims=False, name=None):
    """Compute the variance value of elements along the given axis.

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype='float32')

    # A negative axis is the last-k axis
    print(tf.math.reduce_variance(x, 1))
    print(tf.math.reduce_variance(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(tf.math.reduce_variance(x))  # 2.917

    # Also, ``axis`` could be a sequence of integers
    print(tf.math.reduce_variance(x, [0, 1]))  # 2.917
    ```

    Parameters
    ----------
    input_tensor : dragon.Tensor
        The input tensor.
    axis : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.var(input_tensor, axis, keepdims=keepdims, name=name)


def round(x, name=None):
    r"""Compute the nearest integer of input.

    .. math:: \text{out} = \lfloor \text{input} \rceil

    Examples:

    ```python
    x = tf.constant([0.9, 1.4, 1.9])
    print(tf.math.round(x))  # [1., 1., 2.]
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

    """
    return math_ops.round(x, name=name)


def rsqrt(x, name=None):
    r"""Compute the reciprocal square root of input.

    .. math:: \text{out} = \frac{1}{\sqrt{\text{input}}}

    Examples:

    ```python
    x = tf.constant([0., 4., 16.])
    print(tf.math.rsqrt(x))  # [inf, 0.5, 0.25]
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

    """
    return math_ops.rsqrt(x, name=name)


def sigmoid(x, name=None, **kwargs):
    r"""Compute the sigmoid function.

    .. math:: \text{out} = \frac{1}{1 + \exp(-\text{input})}

    Examples:

    ```python
    x = tf.constant([0.2, 0.4, 0.6, 0.8, 1.0])
    print(tf.math.sigmoid(x))
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
        The output tensor

    """
    return activation_ops.sigmoid(x, name=name, **kwargs)


def sign(x, name=None):
    r"""Compute the sign indication of input.

    .. math::
        \text{out}_{i} =
            \begin{cases}
                -1, & \text{ if } \text{input}_{i} < 0 \\
                 0, & \text{ if } \text{input}_{i} = 0 \\
                 1, & \text{ if } \text{input}_{i} > 0
            \end{cases}

    Examples:

    ```python
    x = tf.constant([-2, 0, 2])
    print(tf.math.sign(x))  # [-1, 0, 1]
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

    """
    return math_ops.sign(x, name=name)


def sin(x, name=None):
    r"""Compute the sin of input.

    .. math:: \text{out} = \sin(\text{input})

    Examples:

    ```python
    x = tf.constant((0., math.pi / 2))
    print(tf.math.sin(x))  # [0., 1.]
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

    """
    return math_ops.sin(x, name=name)


def sqrt(x, name=None):
    r"""Compute the square root of input.

    .. math:: \text{out} = \sqrt{\text{input}}

    Examples:

    ```python
    x = tf.constant([4., 9., 16.])
    print(tf.math.sqrt(x))  # [2., 3., 4.]
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

    """
    return math_ops.sqrt(x, name=name)


def square(x, name=None):
    r"""Compute the square of input.

    .. math:: \text{out} = \text{input}^{2}

    Examples:

    ```python
    x = tf.constant([2, 3, 4])
    print(tf.math.square(x))  # [4, 9, 16]
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

    """
    return math_ops.square(x, name=name)


def subtract(x, y, name=None):
    r"""Compute the element-wise subtraction.

    .. math:: \text{out} = \text{input1} - \text{input2}

    Examples:

    ```python
    x = tf.constant(1)
    y = tf.constant(2)
    print(tf.math.subtract(x, y))
    print(x - y)  # Equivalent operation
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input1 tensor.
    y : dragon.Tensor
        The input2 tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.sub([x, y], name=name)


def tanh(x, name=None, **kwargs):
    r"""Compute the tanh of input.

    .. math:: \text{out} = \frac{\exp(\text{input}) - \exp(-\text{input})}
                                {\exp(\text{input}) + \exp(-\text{input})}

    Examples:

    ```python
    x = tf.constant([0.2, 0.4, 0.6, 0.8, 1.0])
    print(tf.math.tanh(x))
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

    """
    return activation_ops.tanh(x, name=name, **kwargs)
