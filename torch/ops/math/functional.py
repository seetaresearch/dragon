# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.ops import utils
from dragon.vm.torch.ops.math import _functions


def abs(input, out=None):
    r"""Compute the absolute value of input.

    .. math:: \text{out} = \left| \text{input} \right|

    Examples:

    ```python
    print(torch.abs(torch.tensor([-1, 0, 1])))  # [1, 0, 1]
    ```

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
    return _unary_func(input, 'Abs', out)


def axpby(input, alpha=1., beta=1., out=None):
    r"""Compute the element-wise addition from input to output.

    .. math:: \text{out} = \alpha * \text{input} + \beta * \text{out}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    alpha : float, optional, default=1.
        The value of :math:`\alpha`.
    beta : float, optional, default=1.
        The value of :math:`\beta`.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Axpby \
        .instantiate(
            input.device,
            alpha=alpha,
            beta=beta,
        ).apply(input, out)


def add(input, value, out=None):
    r"""Compute the element-wise addition.

    .. math:: \text{out} = \text{input} + \text{value}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    value : Union[dragon.vm.torch.Tensor, number]
        The value to add.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, value, 'Add', out)


def bitwise_not(input, out=None):
    r"""Compute the element-wise NOT bitwise operation.

    .. math:: \text{out} = \,\,\sim x

    Examples:

    ```python
    # Typically, ``x`` is a bool tensor
    print(torch.bitwise_nor(torch.tensor([0, 1], 'bool')))  # [True, False]

    # Otherwise, integral types are required (unsigned or signed)
    # 00001101 (13) -> 11110010 (?)
    print(torch.bitwise_not(torch.tensor(13, 'uint8')))  # 242
    print(torch.bitwise_not(torch.tensor(13, 'int8')))   # -14
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The tensor :math:`x`.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _unary_func(input, 'Invert', out)


def bitwise_xor(input, other, out=None):
    r"""Compute the element-wise XOR bitwise operation.

    .. math:: \text{out} = \text{input} \oplus \text{other}

    Examples:

    ```python
    a = torch.tensor([False, True, False, True])
    b = torch.tensor([False, True, True, False])
    print(torch.bitwise_xor(a, b))  # False, False, True, True
    print(a - b)  # Equivalent operation
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The first input tensor.
    other : dragon.vm.torch.Tensor
        The second input tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Sub', out)


def ceil(input, out=None):
    r"""Compute the smallest integer not less than input.

    .. math:: \text{out} = \lceil x \rceil

    Examples:

    ```python
    x = torch.tensor([1.4, 1.7, 2.0])
    print(torch.ceil(x))  # [2., 2., 2.]
    ```

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
    return _unary_func(input, 'Ceil', out)


def clamp(input, min=None, max=None, out=None):
    r"""Compute the clipped input according to the given bounds.

    .. math:: \text{out} = \min(\max(\text{input}, low), high)

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    min : number, optional
        The min value.
    max : number, optional
        The max value.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Clip \
        .instantiate(
            input.device,
            min=float(min) if min is not None else None,
            max=float(max) if max is not None else None,
        ).apply(input, out)


def cos(input, out=None):
    r"""Compute the cos of input.

    .. math:: \text{out} = \cos(\text{input})

    Examples:

    ```python
    x = torch.tensor([0., math.pi])
    print(torch.cos(x))  # [1., -1.]
    ```

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
    return _unary_func(input, 'Cos', out)


def div(input, value, out=None):
    r"""Compute the element-wise division.

    .. math:: \text{out} = \text{input} \div \text{value}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    value : Union[dragon.vm.torch.Tensor, number]
        The value to divide.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, value, 'Div', out)


def eq(input, other, out=None):
    r"""Compute the element-wise equal comparison.

    .. math:: \text{out} = (\text{input} = \text{other})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Equal', out)


def exp(input, out=None):
    r"""Compute the exponential of input.

    .. math:: \text{out} = e^{\text{input}}

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
    return _unary_func(input, 'Exp', out)


def floor(input, out=None):
    r"""Compute the largest integer not greater than input.

    .. math:: \text{out} = \lfloor x \rfloor

    Examples:

    ```python
    x = torch.tensor([0.9, 1.4, 1.9])
    print(torch.floor(x))  # [0., 1., 1.]
    ```

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
    return _unary_func(input, 'Floor', out)


def ge(input, other, out=None):
    r"""Compute the element-wise greater-equal comparison.

    .. math:: \text{out} = (\text{input} \geq \text{other})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'GreaterEqual', out)


def gt(input, other, out=None):
    r"""Compute the element-wise greater comparison.

    .. math:: \text{out} = (\text{input} > \text{other})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _binary_func(input, other, 'Greater', out)


def isinf(input):
    r"""Check if the elements of input are infinite.

    .. math:: \text{out} = \text{isinf}(x)

    Examples:

    ```python
    x = torch.tensor([0., 1., float('inf')])
    print(torch.isinf(x))  # [False, False, True]
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
    return _unary_func(input, 'IsInf')


def isnan(input):
    r"""Check if the elements of input are NaN.

    .. math:: \text{out} = \text{isnan}(x)

    Examples:

    ```python
    x = torch.tensor([0., 1., float('nan')])
    print(torch.isnan(x))  # [False, False, True]
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
    return _unary_func(input, 'IsNaN')


def le(input, other, out=None):
    r"""Compute the element-wise less-equal comparison.

    .. math:: \text{out} = (\text{input} \leq \text{other})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _binary_func(input, other, 'LessEqual', out)


def log(input, out=None):
    r"""Compute the natural logarithm of input.

    .. math:: \text{out} = \log(\text{input})

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
    return _unary_func(input, 'Log', out)


def logsumexp(input, dim, keepdim=False, out=None):
    r"""Apply the composite of log, sum, and exp to input.

    .. math:: \text{LogSumExp}(x)_{i} = \log\sum_{j}\exp(x_{ij})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]]
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Whether the output tensor has dim retained or not.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return log(exp(input).sum(dim, keepdim), out)


def lt(input, other, out=None):
    r"""Compute the element-wise less comparison.

    .. math:: \text{out} = (\text{input} < \text{other})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _binary_func(input, other, 'Less', out)


def maximum(input, other, out=None):
    r"""Compute the maximum value of inputs.

    .. math:: \text{out} = \max(\text{input}, \text{other})

    Parameters
    ----------
    input : Union[dragon.vm.torch.Tensor, number]
        The first input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The second input tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    input, other = utils \
        .remove_binary_scalar(input, other)
    return _functions.BinaryFunc \
        .instantiate(
            input.device,
            op_type='Maximum',
        ).apply(input, other, out)


def minimum(input, other, out=None):
    r"""Compute the minimum value of inputs.

    .. math:: \text{out} = \min(\text{input}, \text{other})

    Parameters
    ----------
    input : Union[dragon.vm.torch.Tensor, number]
        The first input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The second input tensor.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    input, other = utils \
        .remove_binary_scalar(input, other)
    return _functions.BinaryFunc \
        .instantiate(
            input.device,
            op_type='Minimum',
        ).apply(input, other, out)


def mm(input, mat2, transpose_a=False, transpose_b=False, out=None):
    r"""Compute matrix-matrix multiplication.

    .. math:: \text{out} = a \times b

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The matrix :math:`a`.
    mat2 : dragon.vm.torch.Tensor
        The matrix :math:`b`.
    transpose_a : bool, optional, default=False
        **True** to transpose :math:`a` before computation.
    transpose_b : bool, optional, default=False
        **True** to transpose :math:`b` before computation.
    out : dragon.vm.torch.Tensor, optional
        The optional output.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.MatMul \
        .instantiate(
            utils.unify_devices([input, mat2]),
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        ).apply(input, mat2, out)


def mul(input, value, out=None):
    r"""Compute the element-wise multiplication.

    .. math:: \text{out} = \text{input} \times \text{value}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    value : Union[dragon.vm.torch.Tensor, number]
        The value to multiply.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, value, 'Mul', out)


def ne(input, other, out=None):
    r"""Compute the element-wise not-equal comparison.

    .. math:: \text{out} = (\text{input} \neq \text{other})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _binary_func(input, other, 'NotEqual', out)


def neg(input, out=None):
    r"""Compute the element-wise negative.

    .. math:: \text{out} = -\text{input}

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
    return _unary_func(input, 'Neg', out)


def pow(input, exponent, out=None):
    r"""Compute the power of input.

    .. math:: \text{out} = x^{y}

    The two inputs should be broadcast to each other:

    ```python
    x = torch.tensor([[2, 2]])
    print(torch.pow(x, x))  # [[4, 4]]
    print(torch.pow(x, 3))  # [[8, 8]]
    print(torch.pow(3, x))  # [[9, 9]]
    ```

    Parameters
    ----------
    input : Union[dragon.vm.torch.Tensor, number]
        The input tensor :math:`x`.
    exponent : Union[dragon.vm.torch.Tensor, number]
        The exponent value :math:`y`.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, exponent, 'Pow', out)


def reciprocal(input, out=None):
    r"""Compute the reciprocal of input.

    .. math:: \text{out} = \frac{1}{\text{input}}

    Examples:

    ```python
    x = torch.tensor([0., 1., 2.])
    print(torch.reciprocal(x))  # [inf, 1., 0.5]
    ```

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
    return _unary_func(input, 'Reciprocal', out)


def round(input, out=None):
    r"""Compute the nearest integer of input.

    .. math:: \text{out} = \lfloor \text{input} \rceil

    Examples:

    ```python
    x = torch.tensor([0.9, 1.4, 1.9])
    print(torch.round(x))  # [1., 1., 2.]
    ```

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
    return _unary_func(input, 'Round', out)


def rsqrt(input, out=None):
    r"""Compute the reciprocal square root of input.

    .. math:: \text{out} = \frac{1}{\sqrt{\text{input}}}

    Examples:

    ```python
    x = torch.tensor([0., 4., 16.])
    print(torch.rsqrt(x))  # [inf, 0.5, 0.25]
    ```

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
    return _unary_func(input, 'Rsqrt', out)


def sign(input, out=None):
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
    x = torch.tensor([-2, 0, 2])
    print(torch.sign(x))  # [-1, 0, 1]
    ```

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
    return _unary_func(input, 'Sign', out)


def sin(input, out=None):
    r"""Compute the sin of input.

    .. math:: \text{out} = \sin(\text{input})

    Examples:

    ```python
    x = torch.tensor([0., math.pi / 2])
    print(torch.sin(x))  # [0., 1.]
    ```

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
    return _unary_func(input, 'Sin', out)


def sqrt(input, out=None):
    r"""Compute the square root of input.

    .. math:: \text{out} = \sqrt{\text{input}}

    Examples:

    ```python
    x = torch.tensor([4., 9., 16.])
    print(torch.sqrt(x))  # [2., 3., 4.]
    ```

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
    return _unary_func(input, 'Sqrt', out)


def sub(input, value, out=None):
    r"""Compute the element-wise subtraction.

    .. math:: \text{out} = \text{input} - \text{value}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    value : Union[dragon.vm.torch.Tensor, number]
        The value to subtract.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, value, 'Sub', out)


def _binary_func(input, value, op_type='', out=None):
    """Generic binary function."""
    input, value = utils.remove_binary_scalar(input, value)
    return _functions.BinaryFunc \
        .instantiate(input.device, op_type=op_type) \
        .apply(input, value, out)


def _unary_func(input, op_type='', out=None):
    """Generic unary function."""
    return _functions.UnaryFunc \
        .instantiate(input.device, op_type=op_type) \
        .apply(input, out)
