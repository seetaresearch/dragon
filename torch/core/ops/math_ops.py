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

from dragon.core.util import nest
from dragon.vm.torch.core.autograd.function import Function
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
    return Function.apply(
        'Gemm', input.device, [mat1, mat2, input], outputs=[out],
        alpha=float(alpha), beta=float(beta))


def argmax(input, dim, keepdim=False, out=None):
    """Return the index of maximum elements along the given dimension.

    :attr:`dim` could be negative:

    ```python
    # A negative dimension is the last-k dimension
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(torch.argmax(x, dim=1))
    print(torch.argmax(x, dim=-1))  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The index of maximum elements.

    """
    return Function.apply(
        'ArgMax', input.device, [input], outputs=[out],
        axis=dim, keepdims=keepdim)


def argmin(input, dim, keepdim=False, out=None):
    """Return the index of minimum elements along the given dimension.

    :attr:`dim` could be negative:

    ```python
    # A negative dimension is the last-k dimension
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(torch.argmin(x, dim=1))
    print(torch.argmin(x, dim=-1))  # Equivalent
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
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The index of minimum elements.

    """
    return Function.apply(
        'ArgMin', input.device, [input], outputs=[out],
        axis=dim, keepdims=keepdim)


def atan2(input, other, out=None):
    r"""Compute the element-wise arc-tangent of two arguments.

    .. math:: \text{out} = \text{arctan}(\frac{\text{input}}{\text{other}})

    Examples:

    ```python
    y = torch.tensor(1.)
    x = torch.tensor(2.)
    print(torch.atan2(y, x))  # 0.46364761
    ```

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
    return _binary_func(input, other, 'Atan2', out)


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
    return Function.apply(
        'MatMul', input.device, [input, mat2], outputs=[out])


def cast(input, dtype='float32', out=None):
    """Cast the data type of input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input.
    dtype : str, optional, default='float32'
        The data type to cast to.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'Cast', input.device, [input], outputs=[out], dtype=dtype)


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
    return Function.apply(
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


def cumsum(input, dim, out=None):
    """Compute the cumulative sum of elements along the given dimension.

    :attr:`dim` could be negative:

    ```python
    # A negative dimension is the last-k dimension
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(torch.cumsum(x, dim=1))  # [[1, 3, 6], [4, 9, 15]]
    print(torch.cumsum(x, dim=-1))  # Equivalent
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int
        The cumulative dimension.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'CumSum', input.device, [input], outputs=[out], axis=dim)


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


def isfinite(input):
    r"""Check if the elements of input are finite.

    .. math:: \text{out} = \text{isfinite}(\text{input})

    Examples:

    ```python
    x = torch.tensor([0., float('nan'), float('inf')])
    print(torch.isfinite(x))  # [True, False, False]
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
    return _unary_func(input, 'IsFinite')


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
    return Function.apply(
        'MatMul', input.device, [input, other], outputs=[out])


def max(input, dim=None, keepdim=False, out=None):
    """Compute the max value of elements along the given dimension.

    :attr:`dim` could be negative or ``None``:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative dimension is the last-k dimension
    print(torch.max(x, dim=1))
    print(torch.max(x, dim=-1))  # Equivalent

    # If dimension is None, reduce input as a vector
    # and return a scalar result
    print(torch.max(x))  # 6

    # Also, dimension could be a sequence of integers
    print(torch.max(x, (0, 1)))  # 6
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    keepdim = keepdim if dim is not None else False
    dim = nest.flatten(dim) if dim is not None else dim
    return Function.apply(
        'ReduceMax', input.device, [input], outputs=[out],
        axes=dim, keepdims=keepdim)


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


def mean(input, dim=None, keepdim=False, out=None):
    """Compute the mean value of elements along the given dimension.

    :attr:`dim` could be negative or ``None``:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # A negative dimension is the last-k dimension
    print(torch.mean(x, dim=1))
    print(torch.mean(x, dim=-1))  # Equivalent

    # If dimension is None, reduce input as a vector
    # and return a scalar result
    print(torch.mean(x))  # 3.5

    # Also, dimension could be a sequence of integers
    print(torch.mean(x, dim=(0, 1)))  # 3.5
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    keepdim = keepdim if dim is not None else False
    dim = nest.flatten(dim) if dim is not None else dim
    return Function.apply(
        'ReduceMean', input.device, [input], outputs=[out],
        axes=dim, keepdims=keepdim)


def min(input, dim=None, keepdim=False, out=None):
    """Compute the min value of elements along the given dimension.

    :attr:`dim` could be negative or ``None``:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative dimension is the last-k dimension
    print(torch.min(x, dim=1))
    print(torch.min(x, dim=-1))  # Equivalent

    # If dimension is None, reduce input as a vector
    # and return a scalar result
    print(torch.min(x))  # 1

    # Also, dimension could be a sequence of integers
    print(torch.min(x, (0, 1)))  # 1
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    keepdim = keepdim if dim is not None else False
    dim = nest.flatten(dim) if dim is not None else dim
    return Function.apply(
        'ReduceMin', input.device, [input], outputs=[out],
        axes=dim, keepdims=keepdim)


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
    return Function.apply(
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


def norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None):
    """Compute the norm value of elements along the given dimension.

    :attr:`dim` could be negative or ``None``:

    ```python
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

    # A negative dimension is the last-k axis
    print(torch.norm(x, dim=1))
    print(torch.norm(x, dim=-1))  # Equivalent

    # If ``dim`` is None, the vector-style reduction
    # will be applied to return a scalar result
    print(torch.norm(x))  # 9.539

    # Also, ``dim`` could be a sequence of integers
    print(torch.norm(x, dim=(0, 1)))  # 9.539
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    p : {'fro', 1, 2}, optional
        The norm order.
    dim : Union[int, Sequence[int]], optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.
    dtype : str, optional
        The data type to cast to.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if p is None or p == 2 or p == 'fro':
        op_type = 'ReduceL2'
    elif p == 1:
        op_type = 'ReduceL1'
    else:
        raise ValueError('Unsupported norm order: ' + str(p))
    input = input.to(dtype=dtype)
    keepdim = keepdim if dim is not None else False
    dim = nest.flatten(dim) if dim is not None else dim
    return Function.apply(
        op_type, input.device, [input], outputs=[out],
        axes=dim, keepdims=keepdim)


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


def square(input, out=None):
    r"""Compute the square of input.

    .. math:: \text{out} = \text{input}^{2}

    Examples:

    ```python
    x = torch.tensor([2., 3., 4.])
    print(torch.square(x))  # [4., 9., 16.]
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
    return _unary_func(input, 'Square', out)


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


def sum(input, dim=None, keepdim=False, out=None):
    """Compute the sum value of elements along the given dimension.

    :attr:`dim` could be negative or ``None``:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # A negative dimension is the last-k dimension
    print(torch.sum(x, dim=1))
    print(torch.sum(x, dim=-1))  # Equivalent

    # If dimension is None, reduce input as a vector
    # and return a scalar result
    print(torch.sum(x))  # 21

    # Also, dimension could be a sequence of integers
    print(torch.sum(x, (0, 1)))  # 21
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    keepdim = keepdim if dim is not None else False
    dim = nest.flatten(dim) if dim is not None else dim
    return Function.apply(
        'ReduceSum', input.device, [input], outputs=[out],
        axes=dim, keepdims=keepdim)


def var(input, dim=None, keepdim=False, out=None):
    """Compute the variance value of elements along the given dimension.

    :attr:`dim` could be negative or ``None``:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # A negative dimension is the last-k dimension
    print(torch.var(x, dim=1))
    print(torch.var(x, dim=-1))  # Equivalent

    # If dimension is None, reduce input as a vector
    # and return a scalar result
    print(torch.var(x))  # 2.917

    # Also, dimension could be a sequence of integers
    print(torch.var(x, (0, 1)))  # 2.917
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    keepdim = keepdim if dim is not None else False
    dim = nest.flatten(dim) if dim is not None else dim
    return Function.apply(
        'ReduceVar', input.device, [input], outputs=[out],
        axes=dim, keepdims=keepdim)


def var_mean(input, dim=None, keepdim=False, out=None):
    """Compute the variance and mean of elements along the given dimension.

    :attr:`dim` could be negative or ``None``:

    ```python
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    # A negative dimension is the last-k dimension
    print(torch.var_mean(x, dim=1))
    print(torch.var_mean(x, dim=-1))  # Equivalent

    # If dimension is None, reduce input as a vector
    # and return a scalar result
    print(torch.var_mean(x))  # (2.917, 3.5)

    # Also, dimension could be a sequence of integers
    print(torch.var_mean(x, (0, 1)))  # (2.917, 3.5)
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    out : Sequence[dragon.vm.torch.Tensor], optional
        The optional output value and index.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The variance and mean tensor.

    """
    keepdim = keepdim if dim is not None else False
    dim = nest.flatten(dim) if dim is not None else dim
    return Function.apply(
        'Moments', input.device, [input],
        outputs=out if out else [None, None],
        axes=dim, keepdims=keepdim)[::-1]


def _binary_func(input, value, op_type, out=None):
    """Compute a binary function."""
    input, value = constant_ops.remove_scalars(input, value)
    return Function.apply(
        op_type, input.device, [input, value], outputs=[out])


def _unary_func(input, op_type, out=None):
    """Compute a unary function."""
    return Function.apply(
        op_type, input.device, [input], outputs=[out])
