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

from dragon.vm.torch.core.autograd.function_impl import FunctionLib
from dragon.vm.torch.core.ops import constant_ops


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
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _unary_func(input, 'Abs', out)


def add(input, other, out=None):
    r"""Compute the element-wise addition.

    .. math:: \text{out} = \text{input} + \text{other}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The tensor to add.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Add', out)


def addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
    r"""Add input to the result of matrix-matrix multiplication.

    .. math:: \text{out} = \alpha (\text{mat1} \times \text{mat2}) + \beta \text{input}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    mat1 : dragon.vm.torch.Tensor
        The first matrix.
    mat2 : dragon.vm.torch.Tensor
        The second matrix.
    beta : float, optional, default=1
        The value to :math:`\beta`.
    alpha : float, optional, default=1
        The value to :math:`\alpha`.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return FunctionLib.apply(
        'Gemm', input.device, [mat1, mat2, input], outputs=[out],
        alpha=float(alpha), beta=float(beta))


def baddbmm(input, batch1, batch2, beta=1, alpha=1, out=None):
    r"""Add input to the result of batched matrix-matrix multiplication.

    .. math::
        \text{out}_{i} = \alpha (\text{mat1}_{i} \times \text{mat2}_{i}) +
                         \beta \text{input}_{i}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    batch1 : dragon.vm.torch.Tensor
        The first batch of matrices.
    batch2 : dragon.vm.torch.Tensor
        The second batch of matrices.
    beta : float, optional, default=1
        The value to :math:`\beta`.
    alpha : float, optional, default=1
        The value to :math:`\alpha`.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    input1 = bmm(batch1, batch2)
    input2 = input * beta if beta != 1 else input
    input1 = input1 * alpha if alpha != 1 else input1
    return add(input1, input2, out)


def bitwise_and(input, other, out=None):
    r"""Compute the element-wise AND bitwise operation.

    .. math:: \text{out} = \text{input} \mathbin{\&} \text{other}

    Examples:

    ```python
    a = torch.tensor([0, -1, 2, -3, 4])
    b = torch.tensor([-4, 3, -2, 1, 0])
    print(torch.bitwise_and(a, b))  # [0, 3, 2, 1, 0]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The first input tensor.
    other : dragon.vm.torch.Tensor
        The second input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'BitwiseAnd', out)


def bitwise_not(input, out=None):
    r"""Compute the element-wise NOT bitwise operation.

    .. math:: \text{out} = \,\,\sim \text{input}

    Examples:

    ```python
    # Typically, ``x`` is a bool tensor
    print(torch.bitwise_not(torch.tensor([0, 1], 'bool')))  # [True, False]

    # Otherwise, integral types are required (unsigned or signed)
    # 00001101 (13) -> 11110010 (?)
    print(torch.bitwise_not(torch.tensor(13, 'uint8')))  # 242
    print(torch.bitwise_not(torch.tensor(13, 'int8')))   # -14
    ```

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
    return _unary_func(input, 'BitwiseNot', out)


def bitwise_or(input, other, out=None):
    r"""Compute the element-wise OR bitwise operation.

    .. math:: \text{out} = \text{input} \mathbin{|} \text{other}

    Examples:

    ```python
    a = torch.tensor([0, -1, 2, -3, 4])
    b = torch.tensor([-4, 3, -2, 1, 0])
    print(torch.bitwise_or(a, b))  # [-4, -1, -2, -3, 4]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The first input tensor.
    other : dragon.vm.torch.Tensor
        The second input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'BitwiseOr', out)


def bitwise_xor(input, other, out=None):
    r"""Compute the element-wise XOR bitwise operation.

    .. math:: \text{out} = \text{input} \oplus \text{other}

    Examples:

    ```python
    a = torch.tensor([0, -1, 2, -3, 4])
    b = torch.tensor([-4, 3, -2, 1, 0])
    print(torch.bitwise_xor(a, b))  # [-4, -4, -4, -4, 4]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The first input tensor.
    other : dragon.vm.torch.Tensor
        The second input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'BitwiseXor', out)


def bmm(input, mat2, out=None):
    r"""Compute the batched matrix-matrix multiplication.

    .. math:: \text{out}_{i} = \text{input}_{i} \times \text{mat2}_{i}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The first batch of matrices.
    mat2 : dragon.vm.torch.Tensor
        The second batch of matrices.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return FunctionLib.apply(
        'MatMul', input.device, [input, mat2], outputs=[out])


def ceil(input, out=None):
    r"""Compute the smallest integer not less than input.

    .. math:: \text{out} = \lceil \text{input} \rceil

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
        The output tensor.

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
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    low = float(min) if min is not None else None
    high = float(max) if max is not None else None
    return FunctionLib.apply(
        'Clip', input.device, [input], outputs=[out], low=low, high=high)


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
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _unary_func(input, 'Cos', out)


def div(input, other, out=None):
    r"""Compute the element-wise division.

    .. math:: \text{out} = \text{input} \div \text{other}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The tensor to divide.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Div', out)


def eq(input, other, out=None):
    r"""Compute the element-wise equal comparison.

    .. math:: \text{out} = (\text{input} == \text{other})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The tensor to compare.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Equal', out)


def exp(input, out=None):
    r"""Compute the exponential of input.

    .. math:: \text{out} = \exp(\text{input})

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
    return _unary_func(input, 'Exp', out)


def floor(input, out=None):
    r"""Compute the largest integer not greater than input.

    .. math:: \text{out} = \lfloor \text{input} \rfloor

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
        The output tensor.

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
        The tensor to compare.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

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
        The tensor to compare.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _binary_func(input, other, 'Greater', out)


def isinf(input):
    r"""Check if the elements of input are infinite.

    .. math:: \text{out} = \text{isinf}(\text{input})

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

    .. math:: \text{out} = \text{isnan}(\text{input})

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
        The tensor to compare.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

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
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _unary_func(input, 'Log', out)


def logical_and(input, other, out=None):
    r"""Compute the element-wise AND logical operation.

    .. math:: \text{out} = \text{input} \mathbin{\&} \text{other}

    Examples:

    ```python
    a = torch.tensor([False, True, False, True])
    b = torch.tensor([False, True, True, False])
    c = torch.Tensor([0, 1, 0, 2])
    d = torch.Tensor([0, 3, 4, 0])
    print(torch.logical_and(a, b))  # [False, True, False, False]
    print(torch.logical_and(c, d))  # [False, True, False, False]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The first input tensor.
    other : dragon.vm.torch.Tensor
        The second input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'And', out)


def logical_not(input, out=None):
    r"""Compute the element-wise NOT logical operation.

    .. math:: \text{out} = \,\,\sim \text{input}

    Examples:

    ```python
    a = torch.tensor([False, True, True])
    b = torch.tensor([0, 1, 2])
    print(torch.logical_not(a))  # [True, False, False]
    print(torch.logical_not(b))  # [True, False, False]
    ```

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
    return _unary_func(input, 'Not', out)


def logical_or(input, other, out=None):
    r"""Compute the element-wise OR logical operation.

    .. math:: \text{out} = \text{input} \mathbin{|} \text{other}

    Examples:

    ```python
    a = torch.tensor([False, True, False, True])
    b = torch.tensor([False, True, True, False])
    c = torch.Tensor([0, 1, 0, 2])
    d = torch.Tensor([0, 3, 4, 0])
    print(torch.logical_or(a, b))  # [False, True, True, True]
    print(torch.logical_or(c, d))  # [False, True, True, True]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The first input tensor.
    other : dragon.vm.torch.Tensor
        The second input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Or', out)


def logical_xor(input, other, out=None):
    r"""Compute the element-wise XOR logical operation.

    .. math:: \text{out} = \text{input} \oplus \text{other}

    Examples:

    ```python
    a = torch.tensor([False, True, False, True])
    b = torch.tensor([False, True, True, False])
    c = torch.Tensor([0, 1, 0, 2])
    d = torch.Tensor([0, 3, 4, 0])
    print(torch.logical_xor(a, b))  # [False, False, True, True]
    print(torch.logical_xor(c, d))  # [False, False, True, True]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The first input tensor.
    other : dragon.vm.torch.Tensor
        The second input tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Xor', out)


def logsumexp(input, dim, keepdim=False, out=None):
    r"""Apply the composite of log, sum, and exp to input.

    .. math:: \text{out}_{i} = \log\sum_{j}\exp(\text{input}_{ij})

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
        The tensor to compare.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output byte tensor.

    """
    return _binary_func(input, other, 'Less', out)


def matmul(input, other, out=None):
    r"""Compute the matrix multiplication.

    .. math:: \text{out} = \text{input} \times \text{other}

    The behavior depends on the shape of input tensors:

    * If both tensors are 1d, computes the vector product.
    * If tensors are 1d and >=2d, computes the vector-matrix multiplication.
    * If tensors are >=2d and 1d, computes the matrix-vector multiplication.
    * If both tensors are >= 2d, computes the matrix-matrix multiplication.
    * If one tensor is >= 3d, applies batching and broadcasting to the computation.

    Examples:

    ```python
    # Vector x Vector
    a = torch.ones(2)
    b = torch.ones(2)
    print(torch.matmul(a, b))
    # Vector x Matrix
    a = torch.ones(2)
    b = torch.ones(2, 3)
    print(torch.matmul(a, b))
    # Matrix x Vector
    a = torch.ones(3, 2)
    b = torch.ones(2)
    print(torch.matmul(a, b))
    # Matrix x Matrix
    a = torch.ones(2, 3)
    b = torch.ones(3, 2)
    print(torch.matmul(a, b))
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : dragon.vm.torch.Tensor
        The tensor to multiply.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return FunctionLib.apply(
        'MatMul', input.device, [input, other], outputs=[out])


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
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Maximum', out)


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
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Minimum', out)


def mm(input, mat2, out=None):
    r"""Compute the matrix-matrix multiplication.

    .. math:: \text{out} = \text{input} \times \text{mat2}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The first matrix.
    mat2 : dragon.vm.torch.Tensor
        The second matrix.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return FunctionLib.apply(
        'Gemm', input.device, [input, mat2], outputs=[out])


def mul(input, other, out=None):
    r"""Compute the element-wise multiplication.

    .. math:: \text{out} = \text{input} \times \text{other}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The tensor to multiply.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Mul', out)


def ne(input, other, out=None):
    r"""Compute the element-wise not-equal comparison.

    .. math:: \text{out} = (\text{input} \neq \text{other})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The tensor to compare.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

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
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _unary_func(input, 'Neg', out)


def pow(input, exponent, out=None):
    r"""Compute the power of input.

    .. math:: \text{out} = \text{input}^{\text{exponent}}

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
        The input tensor.
    exponent : Union[dragon.vm.torch.Tensor, number]
        The exponent tensor.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

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
        The output tensor.

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
        The output tensor.

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
        The output tensor.

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
        The output tensor.

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
        The output tensor.

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
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _unary_func(input, 'Sqrt', out)


def sub(input, other, out=None):
    r"""Compute the element-wise subtraction.

    .. math:: \text{out} = \text{input} - \text{other}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    other : Union[dragon.vm.torch.Tensor, number]
        The tensor to subtract.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _binary_func(input, other, 'Sub', out)


def _binary_func(input, value, op_type, out=None):
    """Compute a binary function."""
    input, value = constant_ops.remove_scalars(input, value)
    return FunctionLib.apply(
        op_type, input.device, [input, value], outputs=[out])


def _unary_func(input, op_type, out=None):
    """Compute an unary function."""
    return FunctionLib.apply(
        op_type, input.device, [input], outputs=[out])
