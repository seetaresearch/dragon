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
"""Bind tensor methods executed with backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.ops.array import functional as array_funcs
from dragon.vm.torch.ops.math import functional as math_funcs
from dragon.vm.torch.ops.init import functional as init_funcs
from dragon.vm.torch import executor
from dragon.vm.torch.tensor import Tensor


def abs(self):
    r"""Return a tensor with the absolute value.

    .. math:: \text{out} = \left| \text{self} \right|

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.abs(...)`_ : Compute the absolute value of input.

    """
    return math_funcs.abs(self)


def add(self, other):
    r"""Compute the element-wise addition.

    .. math:: \text{out} = \text{self} + \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to add.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.add(...)`_ : Compute the element-wise addition.

    """
    return math_funcs.add(self, other)


def add_(self, other):
    r"""Compute the element-wise addition.

    .. math:: \text{self} \mathrel{+}= \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to add.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.add(...)`_ : Compute the element-wise addition.

    """
    return math_funcs.add(self, other, self)


def backward(self, gradient=None, retain_graph=False):
    """Compute the derivatives of this tensor w.r.t. graph leaves.

    Parameters
    ----------
    gradient : dragon.vm.torch.Tensor, optional
        The optional gradient of this tensor.
    retain_graph : bool, optional, default=False
        **False** to free the graph used to compute grad.

    """
    return executor.run_backward(
        tensors=[self],
        grad_tensors=None if gradient is None else [gradient],
        retain_graph=retain_graph,
    )


def bitwise_not(self):
    r"""Compute the element-wise NOT bitwise operation.

    .. math:: \text{out} = \,\,\sim \text{self}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_not(...)`_ : Compute the element-wise NOT bitwise operation.

    """
    return math_funcs.bitwise_not(self)


def bitwise_not_(self):
    r"""Compute the element-wise NOT bitwise operation.

    .. math:: \text{self} = \,\,\sim \text{self}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_not(...)`_ : Compute the element-wise NOT bitwise operation.

    """
    return math_funcs.bitwise_not(self, self)


def bitwise_xor(self, other):
    r"""Compute the element-wise XOR bitwise operation.

    .. math:: \text{out} = \text{self} \oplus \text{other}

    Parameters
    ----------
    other : dragon.vm.torch.Tensor
        The second input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_xor(...)`_ : Compute the element-wise XOR bitwise operation.

    """
    return math_funcs.bitwise_xor(self, other)


def bitwise_xor_(self, other):
    r"""Compute the element-wise XOR bitwise operation.

    .. math:: \text{self} = \text{self} \oplus \text{other}

    Parameters
    ----------
    other : dragon.vm.torch.Tensor
        The second input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.bitwise_xor(...)`_ : Compute the element-wise XOR bitwise operation.

    """
    return math_funcs.bitwise_xor(self, other, self)


def bool(self):
    """Return a bool tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.cast(self, 'bool', False)


def bool_(self):
    """Cast to a bool tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.cast(self, 'bool', True)


def byte(self):
    """Return an uint8 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.cast(self, 'uint8', False)


def byte_(self):
    """Cast to an uint8 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.cast(self, 'uint8', True)


def ceil(self):
    r"""Return a tensor taken the ceil of elements.

    .. math:: \text{out} = \lceil \text{self} \rceil

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.ceil(...)`_ : Compute the smallest integer not less than input.

    """
    return math_funcs.ceil(self)


def ceil_(self):
    r"""Set to the ceil of elements.

    .. math:: \text{self} = \lceil \text{self} \rceil

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.ceil(...)`_ : Compute the smallest integer not less than input.

    """
    return math_funcs.ceil(self, self)


def char(self):
    """Return an int8 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.cast(self, 'int8', False)


def char_(self):
    """Cast to an int8 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.cast(self, 'int8', True)


def chunk(self, chunks, dim=0):
    """Split self into several parts along the given dim.

    Parameters
    ----------
    chunks : int
        The number of chunks to split.
    dim : int, optional, default=0
        The dim to split.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The output chunks.

    """
    return array_funcs.chunk(self, chunks, dim)


def clamp(self, min=None, max=None):
    """Return a tensor with elements clamped into a range.

    Parameters
    ----------
    min : number, optional
        The min value.
    max : number, optional
        The max value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.clamp(...)`_ : Compute the clipped input according to the given bounds.

    """
    return math_funcs.clamp(self, min, max)


def clamp_(self, min=None, max=None):
    """Clamp elements into the a range.

    Parameters
    ----------
    min : number, optional
        The min value.
    max : number, optional
        The max value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.clamp(...)`_ : Compute the clipped input according to the given bounds.

    """
    return math_funcs.clamp(self, min, max, self)


def cos(self):
    r"""Compute the cos.

    .. math:: \text{out} = \cos(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.cos(...)`_ : Compute the cos of input.

    """
    return math_funcs.cos(self)


def cumsum(self, dim):
    """Return a tensor with the cumulative sum of elements.

    Parameters
    ----------
    dim : int
        The cumulative dimension.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.cumsum(...)`_ : Compute the cumulative sum of elements along the given axis.

    """
    return array_funcs.cumsum(self, dim)


def div(self, other):
    r"""Compute the element-wise division.

    .. math:: \text{out} = \text{self} \div \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to divide.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.div(...)`_ : Compute the element-wise division.

    """
    return math_funcs.div(self, other)


def div_(self, other):
    r"""Compute the element-wise division.

    .. math:: \text{self} \mathrel{\div}= \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to be divided.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.div(...)`_ : Compute the element-wise division.

    """
    return math_funcs.div(self, other, self)


def double(self):
    """Return a float64 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.cast(self, 'float64', False)


def double_(self):
    """Cast to a float64 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.cast(self, 'float64', True)


def eq(self, other):
    r"""Compute the element-wise equal comparison.

    .. math:: \text{out} = (\text{self} = \text{other})

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.eq(...)`_ : Compute the element-wise equal comparison.

    """
    return math_funcs.eq(self, other)


def exp(self):
    r"""Compute the exponential.

    .. math:: \text{out} = \exp{\text{self}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.exp(...)`_ : Compute the exponential of input.

    """
    return math_funcs.exp(self)


def expand(self, *sizes):
    """Return a tensor with elements broadcast.

    Parameters
    ----------
    sizes : int...
        The output dimensions to broadcast to.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.expand(...)`_ : Broadcast input according to given sizes.

    """
    return array_funcs.expand(self, sizes)


def fill_(self, value):
    r"""Fill self with a scalar value.

    .. math:: \text{self} \leftarrow \text{value}

    Parameters
    ----------
    value : number
        The value to fill.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return init_funcs.fill(self, self.shape, value)


def float(self):
    """Return a float32 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.cast(self, 'float32', False)


def float_(self):
    """Cast to a float32 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.cast(self, 'float32', True)


def floor(self):
    r"""Return a tensor taken the floor of elements.

    .. math:: \text{out} = \lfloor \text{self} \rfloor

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.floor(...)`_ : Compute the largest integer not greater than input.

    """
    return math_funcs.floor(self)


def floor_(self):
    r"""Set to the floor of elements.

    .. math:: \text{self} = \lfloor \text{self} \rfloor

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.floor(...)`_ : Compute the largest integer not greater than input.

    """
    return math_funcs.floor(self, self)


def ge(self, other):
    r"""Compute the element-wise greater-equal comparison.

    .. math:: \text{out} = (\text{self} \geq \text{other})

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.ge(...)`_ : Compute the element-wise greater-equal comparison.

    """
    return math_funcs.ge(self, other)


def getitem(self, item):
    """Select elements at the specific index.

    Parameters
    ----------
    item : Union[slice, int, dragon.vm.torch.Tensor]
        The index.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if isinstance(item, Tensor):
        return self.masked_select(item)
    else:
        starts, sizes = _process_indices(item)
        return array_funcs.slice(self, starts, sizes)


def gt(self, other):
    r"""Compute the element-wise greater comparison.

    .. math:: \text{out} = (\text{self} > \text{other})

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.gt(...)`_ : Compute the element-wise greater comparison.

    """
    return math_funcs.gt(self, other)


def half(self):
    """Return a float16 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.cast(self, 'float16', False)


def half_(self):
    """Cast to a float16 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.cast(self, 'float16', True)


def index_select(self, dim, index):
    """Select the elements along the dim dim using index.

    Parameters
    ----------
    dim : Union[int, Sequence[int]]
        The dim(s) to select.
    index : dragon.vm.torch.Tensor
        The indices.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.index_select(self, dim, index)


def _int(self):
    """Return an int32 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.cast(self, 'int32', False)


def _int_(self):
    """Cast to an int32 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.cast(self, 'int32', True)


def le(self, other):
    r"""Compute the element-wise less-equal comparison.

    .. math:: \text{out} = (\text{self} \leq \text{other})

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.le(...)`_ : Compute the element-wise less-equal comparison.

    """
    return math_funcs.le(self, other)


def log(self):
    r"""Compute the natural logarithm.

    .. math:: \text{out} = \log(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_funcs.log(self)


def logsumexp(self, dim, keepdim=False):
    r"""Apply the composite of log, sum, and exp.

    .. math:: \text{LogSumExp}(x)_{i} = \log\sum_{j}\exp(x_{ij})

    Parameters
    ----------
    dim : Union[int, Sequence[int]]
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Whether the output tensor has dim retained or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_funcs.logsumexp(self, dim, keepdim)


def long(self):
    """Return an int64 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.cast(self, 'int64', False)


def long_(self):
    """Cast to an int64 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.cast(self, 'int64', True)


def lt(self, other):
    r"""Compute the element-wise less comparison.

    .. math:: \text{out} = (\text{self} < \text{other})

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.lt(...)`_ : Compute the element-wise less comparison.

    """
    return math_funcs.lt(self, other)


def masked_fill_(self, mask, value):
    r"""Fill self with the value where mask is 1.

    .. math::
        \text{self}[i] =
            \begin{cases}
                \text{value}[i], & \text{ if } \text{mask}[i] = 1 \\
                \text{self}[i], & \text{ otherwise }
        \end{cases}

    Parameters
    ----------
    mask : dragon.vm.torch.Tensor
        The boolean mask.
    value : number
        The value to fill.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.masked_fill(self, mask, value)


def masked_select(self, mask):
    """Select the elements where mask is **1**.

    Parameters
    ----------
    mask : dragon.vm.torch.Tensor
        The mask for selecting.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.masked_select(self, mask)


def max(self, dim=None, keepdim=False):
    """Compute the max value of elements along the given axis.

    Parameters
    ----------
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.max(self, dim, keepdim)


def mean(self, dim=None, keepdim=False):
    """Compute the mean value of elements along the given axis.

    Parameters
    ----------
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.mean(self, dim, keepdim)


def min(self, dim=None, keepdim=False):
    """Compute the min value of elements along the given axis.

    Parameters
    ----------
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.min(self, dim, keepdim)


def mul(self, other):
    r"""Compute the element-wise multiplication.

    .. math:: \text{out} = \text{self} \times \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to multiply.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.mul(...)`_ : Compute the element-wise multiplication.

    """
    return math_funcs.mul(self, other)


def mul_(self, other):
    r"""Compute the element-wise multiplication.

    .. math:: \text{self} \mathrel{\times}= \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to multiply.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.mul(...)`_ : Compute the element-wise multiplication.

    """
    return math_funcs.mul(self, other, self)


def multinomial(self, num_samples, eps=0.):
    """Return a tensor where each row contains ``num_samples``,
    sampled from the multinomial distribution.

    Parameters
    ----------
    num_samples : int
        The number of samples.
    eps : float, optional, default=0.
        The prob to a uniform sampling.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.multinomial(self, num_samples, eps)


def narrow(self, dimension, start, length):
    """Return a new tensor that is a narrowed version of input tensor.

    Parameters
    ----------
    dimension : int
        The dimension to narrow.
    start : int
        The starting position.
    length : int
        The distance to the ending position.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.narrow(self, dimension, start, length)


def ne(self, other):
    r"""Compute the element-wise not-equal comparison.

    .. math:: \text{out} = (\text{self} \neq \text{other})

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.ne(...)`_ : Compute the element-wise not-equal comparison.

    """
    return math_funcs.ne(self, other)


def neg(self):
    r"""Compute the element-wise negative.

    .. math:: \text{out} = -\text{self}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.neg(...)`_ : Compute the element-wise negative.

    """
    return math_funcs.neg(self)


def nonzero(self):
    """Return the indices of non-zero elements.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.nonzero(self)


def normal_(self, mean=0, std=1):
    r"""Fill self from a normal distribution.

    .. math:: \text{self} \sim \mathcal{N}(\mu, \sigma)

    Parameters
    ----------
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return init_funcs.normal_fill(self, mean, std)


def permute(self, *dims):
    """Return a new tensor with the specific order of dimensions.

    Parameters
    ----------
    dims : int...
        The new order of dimensions.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.permute(self, dims)


def pow(self, exponent):
    r"""Compute the power.

    Parameters
    ----------
    exponent : Union[dragon.vm.torch.Tensor, number]
            The exponent value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.pow(...)`_ : Compute the power of input.

    """
    return math_funcs.pow(self, exponent)


def reciprocal(self):
    r"""Compute the reciprocal.

    .. math:: \text{out} = \frac{1}{\text{self}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.reciprocal(...)`_ : Compute the reciprocal of input.

    """
    return math_funcs.reciprocal(self)


def reciprocal_(self):
    r"""Compute the reciprocal.

    .. math:: \text{self} = \frac{1}{\text{self}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.reciprocal(...)`_ : Compute the reciprocal of input.

    """
    return math_funcs.reciprocal(self, self)


def repeat(self, *sizes):
    """Repeat elements along the specified dimensions.

    Parameters
    ----------
    sizes : int...
        The number of times to repeat.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.repeat(self, sizes)


def reshape(self, shape):
    """Return a tensor with the same data but a different shape.

    Parameters
    ----------
    shape : Sequence[int]
        The new shape.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.reshape(...)`_ : Change the shape of input.

    """
    return array_funcs.reshape(self, shape)


def reshape_(self, shape):
    """Change into a new shape with the same data.

    Parameters
    ----------
    shape : Sequence[int]
        The new shape.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.reshape(...)`_ : Change the shape of input.

    """
    return array_funcs.reshape(self, shape, self)


def round(self):
    r"""Return a tensor taken the round of elements.

    .. math:: \text{out} = \lfloor \text{self} \rceil

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.round(...)`_ : Compute the nearest integer of input.

    """
    return math_funcs.round(self)


def round_(self):
    r"""Set to the round of elements.

    .. math:: \text{self} = \lfloor \text{self} \rceil

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.round(...)`_ : Compute the nearest integer of input.

    """
    return math_funcs.round(self, self)


def rsqrt(self):
    r"""Compute the reciprocal square root.

    .. math:: \text{out} = \frac{1}{\sqrt{\text{self}}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.rsqrt(...)`_ : Compute the square root of input.

    """
    return math_funcs.rsqrt(self)


def rsqrt_(self):
    r"""Compute the reciprocal square root.

    .. math:: \text{self} = \frac{1}{\sqrt{\text{self}}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.rsqrt(...)`_ : Compute the square root of input.

    """
    return math_funcs.rsqrt(self, self)


def setitem(self, key, value):
    """Set elements at the specific index.

    Parameters
    ----------
    key : Union[slice, int, dragon.vm.torch.Tensor]
        The index.
    value : Union[dragon.vm.torch.Tensor, number]
        The value to set.

    """
    if isinstance(key, Tensor):
        return self.masked_fill_(key, value)
    else:
        starts, sizes = _process_indices(key)
        return array_funcs.assign(self, starts, sizes, value)


def sign(self):
    r"""Return a tensor taken the sign indication of elements.

    .. math::
        \text{out}[i] =
            \begin{cases}
                -1, & \text{ if } \text{self}[i] < 0 \\
                 0, & \text{ if } \text{self}[i] = 0 \\
                 1, & \text{ if } \text{self}[i] > 0
            \end{cases}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.sign(...)`_ : Compute the sign indication of input.

    """
    return math_funcs.sign(self)


def sign_(self):
    r"""Set to the sign indication of elements.

    .. math::
        \text{self}[i] =
            \begin{cases}
                -1, & \text{ if } \text{self}[i] < 0 \\
                 0, & \text{ if } \text{self}[i] = 0 \\
                 1, & \text{ if } \text{self}[i] > 0
            \end{cases}

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.sign(...)`_ : Compute the sign indication of input.

    """
    return math_funcs.sign(self, self)


def sin(self):
    r"""Compute the sin.

    .. math:: \text{out} = \sin(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.sin(...)`_ : Compute the sin of input.

    """
    return math_funcs.sin(self)


def sqrt(self):
    r"""Compute the square root.

    .. math:: \text{out} = \sqrt{\text{self}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.sqrt(...)`_ : Compute the square root of input.

    """
    return math_funcs.sqrt(self)


def sqrt_(self):
    r"""Compute the square root.

    .. math:: \text{self} = \sqrt{\text{self}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.sqrt(...)`_ : Compute the square root of input.

    """
    return math_funcs.sqrt(self, self)


def squeeze(self, dim=None):
    """Return a tensor with dimensions of size 1 removed.

    Parameters
    ----------
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to remove.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.squeeze(self, dim)


def squeeze_(self, dim=None):
    """Inplace version of ``Tensor.squeeze()``.

    Parameters
    ----------
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to remove.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.squeeze(self, dim, self)


def sum(self, dim=None, keepdim=False):
    """Compute the sum value of elements along the given axis.

    Parameters
    ----------
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimensions or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.sum(self, dim, keepdim)


def sub(self, other):
    r"""Compute the element-wise subtraction.

    .. math:: \text{out} = \text{self} - \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to subtract.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.sub(...)`_ : Compute the element-wise subtraction.

    """
    return math_funcs.sub(self, other)


def sub_(self, other):
    r"""Compute the element-wise subtraction.

    .. math:: \text{self} \mathrel{-}= \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to be subtracted.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    See Also
    --------
    `torch.sub(...)`_ : Compute the element-wise subtraction.

    """
    return math_funcs.sub(self, other, self)


def type(self, dtype=None):
    """Return the data type.

    If ``dtype`` is not **None**, cast ``self`` to the new tensor.

    Parameters
    ----------
    dtype : str, optional
        The specified type.

    Returns
    -------
    Union[str, dragon.vm.torch.Tensor]
        The data type or new tensor.

    """
    if dtype is None:
        return self.dtype
    return array_funcs.cast(self, dtype, True)


def uniform_(self, low=0, high=1):
    r"""Fill self from a uniform distribution.

    .. math:: \text{self} \sim \mathcal{U}(\alpha, \beta)

    Parameters
    ----------
    low : number, optional, default=0
        The value to :math:`\alpha`.
    high : number, optional, default=1
        The value to :math:`\beta`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return init_funcs.uniform_fill(self, low, high)


def unsqueeze(self, dim):
    """Return a tensor with dimensions of size 1 inserted.

    Parameters
    ----------
    dim : Union[int, Sequence[int]]
        The dimensions(s) to insert.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.unsqueeze(self, dim)


def unsqueeze_(self, dim):
    """In-place version of ``Tensor.unsqueeze()``.

    Parameters
    ----------
    dim : Union[int, Sequence[int]]
        The dimensions(s) to insert.

    Returns
    -------
    dragon.vm.torch.Tensor
        The self.

    """
    return array_funcs.unsqueeze(self, dim, self)


def where(self, condition, y):
    r"""Select the elements from two branches under the condition.

    .. math::
        \text{out}[i] =
            \begin{cases}
                \text{self}[i] & \text{ if } cond[i] \text{ is True } \\
                y[i], & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    condition : dragon.vm.torch.Tensor
        The condition to select one of the branches.
    y : dragon.vm.torch.Tensor
        The tensor :math:`y`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_funcs.where(condition, self, y)


def _process_indices(item):
    """Process and normalize the indices."""
    if not isinstance(item, (slice, tuple)):
        # >>> value[?]
        if not isinstance(item, int):
            raise ValueError('The index should be a integer.')
        item = (item,)
    if not isinstance(item, tuple):
        # >>> value[?:?]
        item = tuple([item])
    # >>> value[?:?, ?:?, ...]
    starts, sizes = [], []
    for ix, it in enumerate(item):
        if isinstance(it, slice):
            if it.start is None:
                starts.append(0)
            else:
                starts.append(it.start)
            if it.stop is None:
                sizes.append(-1)
            else:
                sizes.append(it.stop - starts[-1])
                if sizes[-1] == 0:
                    raise ValueError(
                        'The starts and ends of axis {} '
                        'can not be equal, got {}:{}.'
                        .format(ix, starts[-1], it.stop)
                    )
            if it.step is not None:
                raise NotImplementedError
        elif isinstance(it, int):
            starts.append(it)
            sizes.append(0)
        else:
            raise TypeError(
                'Unsupported type of indices: {}'
                .format(type(it).__name__)
            )
    return starts, sizes


# Aliases
Tensor.abs = abs
Tensor.add = add
Tensor.add_ = add_
Tensor.backward = backward
Tensor.bitwise_not = bitwise_not
Tensor.bitwise_not_ = bitwise_not_
Tensor.bitwise_xor = bitwise_xor
Tensor.bitwise_xor_ = bitwise_xor_
Tensor.bool = bool
Tensor.bool_ = bool_
Tensor.byte = byte
Tensor.byte_ = byte_
Tensor.ceil = ceil
Tensor.ceil_ = ceil_
Tensor.char = char
Tensor.char_ = char_
Tensor.chunk = chunk
Tensor.clamp = clamp
Tensor.clamp_ = clamp_
Tensor.cos = cos
Tensor.cumsum = cumsum
Tensor.div = div
Tensor.div_ = div_
Tensor.double = double
Tensor.double_ = double_
Tensor.eq = eq
Tensor.exp = exp
Tensor.expand = expand
Tensor.fill_ = fill_
Tensor.float = float
Tensor.float_ = float_
Tensor.floor = floor
Tensor.floor_ = floor_
Tensor.ge = ge
Tensor.gt = gt
Tensor.half = half
Tensor.half_ = half_
Tensor.index_select = index_select
Tensor.int = _int
Tensor.int_ = _int_
Tensor.le = le
Tensor.long = long
Tensor.long_ = long_
Tensor.log = log
Tensor.logsumexp = logsumexp
Tensor.lt = lt
Tensor.masked_fill_ = masked_fill_
Tensor.masked_select = masked_select
Tensor.max = max
Tensor.mean = mean
Tensor.min = min
Tensor.mul = mul
Tensor.mul_ = mul_
Tensor.multinomial = multinomial
Tensor.narrow = narrow
Tensor.ne = ne
Tensor.neg = neg
Tensor.nonzero = nonzero
Tensor.normal_ = normal_
Tensor.permute = permute
Tensor.pow = pow
Tensor.reciprocal = reciprocal
Tensor.reciprocal_ = reciprocal_
Tensor.repeat = repeat
Tensor.reshape = reshape
Tensor.reshape_ = reshape_
Tensor.round = round
Tensor.round_ = round_
Tensor.rsqrt = rsqrt
Tensor.rsqrt_ = rsqrt_
Tensor.sign = sign
Tensor.sign_ = sign_
Tensor.sin = sin
Tensor.sqrt = sqrt
Tensor.sqrt_ = sqrt_
Tensor.squeeze = squeeze
Tensor.squeeze_ = squeeze_
Tensor.sum = sum
Tensor.sub = sub
Tensor.sub_ = sub_
Tensor.type = type
Tensor.uniform_ = uniform_
Tensor.unsqueeze = unsqueeze
Tensor.unsqueeze_ = unsqueeze_
Tensor.where = where
Tensor.__getitem__ = getitem
Tensor.__radd__ = lambda self, value: math_funcs._binary_func(value, self, 'Add')
Tensor.__rmul__ = lambda self, value: math_funcs._binary_func(value, self, 'Mul')
Tensor.__rsub__ = lambda self, value: math_funcs._binary_func(value, self, 'Sub')
Tensor.__rtruediv__ = lambda self, value: math_funcs._binary_func(value, self, 'Div')
Tensor.__setitem__ = setitem
