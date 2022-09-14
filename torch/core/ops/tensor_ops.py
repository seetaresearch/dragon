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
"""Tensor ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest
from dragon.vm.torch.core.autograd import grad_mode
from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.ops import array_ops
from dragon.vm.torch.core.ops import constant_ops
from dragon.vm.torch.core.ops import math_ops
from dragon.vm.torch.core.ops import sort_ops
from dragon.vm.torch.core.tensor import Tensor


def abs(self):
    r"""Return a tensor with the absolute value.

    .. math:: \text{out} = \left| \text{self} \right|

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.abs(...)`_

    """
    return math_ops.abs(self)


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
    `torch.add(...)`_

    """
    return math_ops.add(self, other)


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
        The output tensor.

    See Also
    --------
    `torch.add(...)`_

    """
    return math_ops.add(self, other, self)


def addmm(self, mat1, mat2, beta=1, alpha=1):
    r"""Add the result of matrix-matrix multiplication.

    .. math:: \text{out} = \alpha (\text{mat1} \times \text{mat2}) + \beta \text{self}

    Parameters
    ----------
    mat1 : dragon.vm.torch.Tensor
        The first matrix.
    mat2 : dragon.vm.torch.Tensor
        The second matrix.
    beta : float, optional, default=1
        The value to :math:`\beta`.
    alpha : float, optional, default=1
        The value to :math:`\alpha`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.addmm(...)`_

    """
    return math_ops.addmm(self, mat1, mat2, beta=beta, alpha=alpha)


def argmax(self, dim, keepdim=False):
    """Return the index of maximum elements.

    Parameters
    ----------
    dim : int
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The index of maximum elements.

    See Also
    --------
    `torch.argmax(...)`_

    """
    return math_ops.argmax(self, dim, keepdim)


def argmin(self, dim, keepdim=False):
    """Return the index of minimum elements.

    Parameters
    ----------
    dim : int
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The index of minimum elements.

    See Also
    --------
    `torch.argmin(...)`_

    """
    return math_ops.argmin(self, dim, keepdim)


def argsort(self, dim=-1, descending=False):
    """Return the index of sorted elements.

    Parameters
    ----------
    dim : int, optional, default=-1
        The dimension to sort elements.
    descending : bool, optional, default=False
        Sort in the descending order or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.argsort(...)`_

    """
    return sort_ops.argsort(self, dim, descending)


def atan2(self, other):
    r"""Compute the element-wise arc-tangent of two arguments.

    .. math:: \text{out} = \text{arctan}(\frac{\text{self}}{\text{other}})

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
    `torch.atan2(...)`_

    """
    return math_ops.atan2(self, other)


def baddbmm(self, batch1, batch2, beta=1, alpha=1):
    r"""Add the result of batched matrix-matrix multiplication.

    .. math::
        \text{out}_{i} = \alpha (\text{batch1}_{i} \times \text{batch2}_{i}) +
                         \beta \text{self}_{i}

    Parameters
    ----------
    batch1 : dragon.vm.torch.Tensor
        The first batch of matrices.
    batch2 : dragon.vm.torch.Tensor
        The second batch of matrices.
    beta : float, optional, default=1
        The value to :math:`\beta`.
    alpha : float, optional, default=1
        The value to :math:`\alpha`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.baddbmm(...)`_

    """
    return math_ops.baddbmm(self, batch1, batch2, beta=beta, alpha=alpha)


def baddbmm_(self, batch1, batch2, beta=1, alpha=1):
    r"""Add the result of batched matrix-matrix multiplication.

    .. math::
        \text{self}_{i} = \alpha (\text{batch1}_{i} \times \text{batch2}_{i}) +
                          \beta \text{self}_{i}

    Parameters
    ----------
    batch1 : dragon.vm.torch.Tensor
        The first batch of matrices.
    batch2 : dragon.vm.torch.Tensor
        The second batch of matrices.
    beta : float, optional, default=1
        The value to :math:`\beta`.
    alpha : float, optional, default=1
        The value to :math:`\alpha`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.baddbmm(...)`_

    """
    return math_ops.baddbmm(
        self, batch1, batch2,
        beta=beta, alpha=alpha, out=self,
    )


def backward(self, gradient=None, retain_graph=False):
    """Compute the derivatives of this tensor w.r.t. graph leaves.

    Parameters
    ----------
    gradient : dragon.vm.torch.Tensor, optional
        The optional gradient of this tensor.
    retain_graph : bool, optional, default=False
        ``False`` to free the graph used to compute grad.

    """
    if not self.requires_grad:
        raise ValueError('Backward from a tensor that does not requires grad.')
    grads = [] if gradient is None else [gradient]
    return Function.backward([self], grads, retain_graph=retain_graph)


def bitwise_and(self, other):
    r"""Compute the element-wise AND bitwise operation.

    .. math:: \text{out} = \text{self} \mathbin{\&} \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_and(...)`_

    """
    return math_ops.bitwise_and(self, other)


def bitwise_and_(self, other):
    r"""Compute the element-wise AND bitwise operation.

    .. math:: \text{self} = \text{self} \mathbin{\&} \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_and(...)`_

    """
    return math_ops.bitwise_and(self, other, self)


def bitwise_not(self):
    r"""Compute the element-wise NOT bitwise operation.

    .. math:: \text{out} = \,\,\sim \text{self}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_not(...)`_

    """
    return math_ops.bitwise_not(self)


def bitwise_not_(self):
    r"""Compute the element-wise NOT bitwise operation.

    .. math:: \text{self} = \,\,\sim \text{self}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_not(...)`_

    """
    return math_ops.bitwise_not(self, self)


def bitwise_or(self, other):
    r"""Compute the element-wise OR bitwise operation.

    .. math:: \text{out} = \text{self} \mathbin{|} \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_or(...)`_

    """
    return math_ops.bitwise_or(self, other)


def bitwise_or_(self, other):
    r"""Compute the element-wise OR bitwise operation.

    .. math:: \text{self} = \text{self} \mathbin{|} \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_or(...)`_

    """
    return math_ops.bitwise_or(self, other, self)


def bitwise_xor(self, other):
    r"""Compute the element-wise XOR bitwise operation.

    .. math:: \text{out} = \text{self} \oplus \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_xor(...)`_

    """
    return math_ops.bitwise_xor(self, other)


def bitwise_xor_(self, other):
    r"""Compute the element-wise XOR bitwise operation.

    .. math:: \text{self} = \text{self} \oplus \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bitwise_xor(...)`_

    """
    return math_ops.bitwise_xor(self, other, self)


def bmm(self, batch2):
    r"""Compute the batched matrix multiplication.

    .. math:: \text{out}_{i} = \text{self}_{i} \times \text{batch2}_{i}

    Parameters
    ----------
    batch2 : dragon.vm.torch.Tensor
        The second batch of matrices.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.bmm(...)`_

    """
    return math_ops.bmm(self, batch2)


def bool(self):
    """Return a bool tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'bool')


def bool_(self):
    """Cast to a bool tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'bool', self)


def byte(self):
    """Return an uint8 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'uint8')


def byte_(self):
    """Cast to an uint8 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'uint8', self)


def ceil(self):
    r"""Return a tensor taken the ceil of elements.

    .. math:: \text{out} = \lceil \text{self} \rceil

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.ceil(...)`_

    """
    return math_ops.ceil(self)


def ceil_(self):
    r"""Set to the ceil of elements.

    .. math:: \text{self} = \lceil \text{self} \rceil

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.ceil(...)`_

    """
    return math_ops.ceil(self, self)


def char(self):
    """Return an int8 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'int8')


def char_(self):
    """Cast to an int8 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'int8', self)


def chunk(self, chunks, dim=0, copy=True):
    """Split self into several parts along the given dim.

    Parameters
    ----------
    chunks : int
        The number of chunks to split.
    dim : int, optional, default=0
        The dimension to split.
    copy : bool, optional, default=True
        Copy or create the views of input.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The output chunks.

    """
    return array_ops.chunk(self, chunks, dim, copy)


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
    `torch.clamp(...)`_

    """
    return math_ops.clamp(self, min, max)


def clamp_(self, min=None, max=None):
    """Clamp elements into the given range.

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
    `torch.clamp(...)`_

    """
    return math_ops.clamp(self, min, max, self)


def cos(self):
    r"""Compute the cos.

    .. math:: \text{out} = \cos(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.cos(...)`_

    """
    return math_ops.cos(self)


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
    `torch.cumsum(...)`_

    """
    return math_ops.cumsum(self, dim)


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
    `torch.div(...)`_

    """
    return math_ops.div(self, other)


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
        The output tensor.

    See Also
    --------
    `torch.div(...)`_

    """
    return math_ops.div(self, other, self)


def double(self):
    """Return a float64 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'float64')


def double_(self):
    """Cast to a float64 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'float64', self)


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
    `torch.eq(...)`_

    """
    return math_ops.eq(self, other)


def exp(self):
    r"""Compute the exponential.

    .. math:: \text{out} = \exp(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.exp(...)`_

    """
    return math_ops.exp(self)


def exp_(self):
    r"""Set to the exponential of elements.

    .. math:: \text{self} = \exp(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.exp(...)`_

    """
    return math_ops.exp(self, self)


def expand(self, *sizes):
    """Return a tensor with elements broadcast.

    Parameters
    ----------
    sizes : Union[Sequence[int], int...]
        The output dimensions to broadcast to.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_ops.broadcast_to(self, sizes)


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
        The output tensor.

    """
    size = self.size()
    return Function.apply(
        'Fill', self.device, [], outputs=[self],
        dtype=self.dtype, value=float(value), ndim=len(size), dims=size)


def flatten(self, start_dim=0, end_dim=-1):
    """Return a tensor with dimensions flattened.

    Parameters
    ----------
    start_dim : int, optional, default=0
        The start dimension to flatten.
    end_dim : int, optional, default=-1
        The end dimension to flatten.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.flatten(...)`_

    """
    return array_ops.flatten(self, start_dim, end_dim)


def flatten_(self, start_dim=0, end_dim=-1):
    """Flatten the dimensions.

    Parameters
    ----------
    start_dim : int, optional, default=0
        The start dimension to flatten.
    end_dim : int, optional, default=-1
        The end dimension to flatten.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.flatten(...)`_

    """
    return array_ops.flatten(self, start_dim, end_dim, self)


def flip(self, dims):
    """Return a tensor with elements reversed along the given dimension.

    Parameters
    ----------
    dims : Union[int, Sequence[int]]
        The dimension to reverse.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.flip(...)`_

    """
    return array_ops.flip(self, dims)


def fliplr(self):
    """Return a tensor with elements reversed along the second dimension.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.fliplr(...)`_

    """
    return array_ops.fliplr(self)


def flipud(self):
    """Return a tensor with elements reversed along the first dimension.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.flipud(...)`_

    """
    return array_ops.flipud(self)


def _float(self):
    """Return a float32 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'float32')


def _float_(self):
    """Cast to a float32 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'float32', self)


def floor(self):
    r"""Return a tensor taken the floor of elements.

    .. math:: \text{out} = \lfloor \text{self} \rfloor

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.floor(...)`_

    """
    return math_ops.floor(self)


def floor_(self):
    r"""Set to the floor of elements.

    .. math:: \text{self} = \lfloor \text{self} \rfloor

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.floor(...)`_

    """
    return math_ops.floor(self, self)


def gather(self, dim, index):
    """Gather elements along the given dimension of index.

    Parameters
    ----------
    dim : int
        The dimension of index values.
    index : dragon.vm.torch.Tensor
        The index tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.gather(...)`_

    """
    return array_ops.gather(self, dim, index)


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
    `torch.ge(...)`_

    """
    return math_ops.ge(self, other)


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
    gather_args = []
    if isinstance(item, Tensor):
        if item.dtype == 'bool' or item.dtype == 'uint8':
            return self.masked_select(item)
        elif item.dtype == 'int64':
            gather_args.append((0, item))
        else:
            raise TypeError('Unsupported index type: ' + item.dtype)
    if isinstance(item, tuple):
        for i, elem in enumerate(item):
            if isinstance(elem, Tensor):
                if elem.dtype == 'int64':
                    gather_args.append((i, elem))
                else:
                    raise TypeError('Unsupported index type: ' + elem.dtype)
    if len(gather_args) == 1:
        return self.index_select(*gather_args[0])
    elif len(gather_args) > 1:
        raise NotImplementedError
    starts, sizes = _process_index(item)
    return Function.apply(
        'Slice', self.device, [self], ndim=len(starts),
        starts=starts, sizes=sizes)


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
    `torch.gt(...)`_

    """
    return math_ops.gt(self, other)


def half(self):
    """Return a float16 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'float16')


def half_(self):
    """Cast to a float16 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'float16', self)


def index_select(self, dim, index):
    """Select the elements along the dim using index.

    Parameters
    ----------
    dim : Union[int, Sequence[int]]
        The dim(s) to select.
    index : dragon.vm.torch.Tensor
        The index.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_ops.index_select(self, dim, index)


def _int(self):
    """Return an int32 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'int32')


def _int_(self):
    """Cast to an int32 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'int32', self)


def isfinite(self):
    r"""Return if the elements are finite.

    .. math:: \text{out} = \text{isfinite}(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.isfinite(...)`_

    """
    return math_ops.isfinite(self)


def isinf(self):
    r"""Return if the elements are infinite.

    .. math:: \text{out} = \text{isinf}(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.isinf(...)`_

    """
    return math_ops.isinf(self)


def isnan(self):
    r"""Return if the elements are NaN.

    .. math:: \text{out} = \text{isnan}(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.isnan(...)`_

    """
    return math_ops.isnan(self)


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
    `torch.le(...)`_

    """
    return math_ops.le(self, other)


def log(self):
    r"""Compute the natural logarithm.

    .. math:: \text{out} = \log(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.log(self)


def log_(self):
    r"""Set to the natural logarithm of elements.

    .. math:: \text{self} = \log(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.log(...)`_

    """
    return math_ops.log(self, self)


def logical_and(self, other):
    r"""Compute the element-wise AND logical operation.

    .. math:: \text{out} = \text{self} \mathbin{\&} \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.logical_and(...)`_

    """
    return math_ops.logical_and(self, other)


def logical_not(self):
    r"""Compute the element-wise NOT logical operation.

    .. math:: \text{out} = \,\,\sim \text{self}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.logical_not(...)`_

    """
    return math_ops.logical_not(self)


def logical_or(self, other):
    r"""Compute the element-wise OR logical operation.

    .. math:: \text{out} = \text{self} \mathbin{|} \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.logical_or(...)`_

    """
    return math_ops.logical_or(self, other)


def logical_xor(self, other):
    r"""Compute the element-wise XOR logical operation.

    .. math:: \text{out} = \text{self} \oplus \text{other}

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.logical_xor(...)`_

    """
    return math_ops.logical_xor(self, other)


def logsumexp(self, dim, keepdim=False):
    r"""Apply the composite of log, sum, and exp.

    .. math:: \text{out}_{i} = \log\sum_{j}\exp(\text{self}_{ij})

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

    See Also
    --------
    `torch.logsumexp(...)`_

    """
    return math_ops.logsumexp(self, dim, keepdim)


def long(self):
    """Return an int64 tensor with the same data.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'int64')


def long_(self):
    """Cast to an int64 tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return math_ops.cast(self, 'int64', self)


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
    `torch.lt(...)`_

    """
    return math_ops.lt(self, other)


def masked_fill(self, mask, value):
    r"""Return a tensor filled with the value where mask is true.

    .. math::
        \text{out}_{i} =
            \begin{cases}
                \text{value}_{i}, & \text{ if } \text{mask}_{i} = 1 \\
                \text{self}_{i}, & \text{ otherwise }
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
        The output tensor.

    """
    return array_ops.masked_fill(self, mask, value)


def masked_fill_(self, mask, value):
    r"""Fill self with the value where mask is 1.

    .. math::
        \text{self}_{i} =
            \begin{cases}
                \text{value}_{i}, & \text{ if } \text{mask}_{i} = 1 \\
                \text{self}_{i}, & \text{ otherwise }
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
        The output tensor.

    """
    return array_ops.masked_fill(self, mask, value, self)


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
    return array_ops.masked_select(self, mask)


def matmul(self, tensor2):
    r"""Compute the matrix multiplication.

    .. math:: \text{out} = \text{self} \times \text{tensor2}

    Parameters
    ----------
    tensor2 : dragon.vm.torch.Tensor
        The tensor to multiply.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.matmul(...)`_

    """
    return math_ops.matmul(self, tensor2)


def max(self, dim=None, keepdim=False):
    """Compute the max value of elements along the given dimension.

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

    See Also
    --------
    `torch.max(...)`_

    """
    return math_ops.max(self, dim, keepdim)


def maximum(self, other):
    r"""Compute the maximum value of inputs.

    .. math:: \text{out} = \max(\text{self}, \text{other})

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The second input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.maximum(...)`_

    """
    return math_ops.maximum(self, other)


def mean(self, dim=None, keepdim=False):
    """Compute the mean value of elements along the given dimension.

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

    See Also
    --------
    `torch.mean(...)`_

    """
    return math_ops.mean(self, dim, keepdim)


def min(self, dim=None, keepdim=False):
    """Compute the min value of elements along the given dimension.

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

    See Also
    --------
    `torch.min(...)`_

    """
    return math_ops.min(self, dim, keepdim)


def minimum(self, other):
    r"""Compute the minimum value of inputs.

    .. math:: \text{out} = \min(\text{self}, \text{other})

    Parameters
    ----------
    other : Union[dragon.vm.torch.Tensor, number]
        The second input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.minimum(...)`_

    """
    return math_ops.minimum(self, other)


def mm(self, mat2):
    r"""Compute the matrix-matrix multiplication.

    .. math:: \text{out} = \text{self} \times \text{mat2}

    Parameters
    ----------
    mat2 : dragon.vm.torch.Tensor
        The second matrix.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.mm(...)`_

    """
    return math_ops.mm(self, mat2)


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
    `torch.mul(...)`_

    """
    return math_ops.mul(self, other)


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
        The output tensor.

    See Also
    --------
    `torch.mul(...)`_

    """
    return math_ops.mul(self, other, self)


def multinomial(self, num_samples):
    """Return a tensor with index sampled from multinomial distribution.

    Parameters
    ----------
    num_samples : int
        The number of samples.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_ops.multinomial(self, num_samples)


def narrow(self, dimension, start, length):
    """Return a narrowed tensor.

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
    return array_ops.narrow(self, dimension, start, length)


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
    `torch.ne(...)`_

    """
    return math_ops.ne(self, other)


def neg(self):
    r"""Compute the element-wise negative.

    .. math:: \text{out} = -\text{self}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.neg(...)`_

    """
    return math_ops.neg(self)


def neg_(self):
    r"""Compute the element-wise negative.

    .. math:: \text{self} = -\text{self}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.neg(...)`_

    """
    return math_ops.neg(self, self)


def new_empty(self, *size, dtype=None, device=None, requires_grad=False):
    """Return a tensor filled with uninitialized data.

    Refer this tensor if ``dtype`` and ``device`` not provided.

    Parameters
    ----------
    size : int...
        The size of output tensor.
    dtype : str, optional
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        ``True`` to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.empty(...)`_

    """
    return constant_ops.empty(
        *nest.flatten(size),
        dtype=self.dtype if dtype is None else dtype,
        device=self.device if device is None else device,
        requires_grad=requires_grad,
    )


def new_full(
    self,
    size,
    fill_value,
    dtype=None,
    device=None,
    requires_grad=False,
):
    """Return a tensor filled with a scalar.

    Refer this tensor if ``dtype`` and ``device`` not provided.

    Parameters
    ----------
    size : Sequence[int]
        The size of output tensor.
    fill_value : number
        The scalar to fill.
    dtype : str, optional
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        ``True`` to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.full(...)`_

    """
    return constant_ops.full(
        size,
        fill_value,
        dtype=self.dtype if dtype is None else dtype,
        device=self.device if device is None else device,
        requires_grad=requires_grad,
    )


def new_tensor(self, data, dtype=None, device=None, requires_grad=False):
    """Return a tensor initializing from the given data.

    Refer this tensor if ``dtype`` and ``device`` not provided.

    Parameters
    ----------
    data : array_like
        The data to initialize from.
    dtype : str, optional
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        ``True`` to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.tensor(...)`_

    """
    return constant_ops.tensor(
        data,
        dtype=self.dtype if dtype is None else dtype,
        device=self.device if device is None else device,
        requires_grad=requires_grad,
    )


def nonzero(self):
    r"""Return the index of non-zero elements.

    .. math:: \text{out} = \{i\}, \text{ if } \text{self}_{i} \neq 0

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nonzero(...)`_

    """
    return array_ops.nonzero(self)


def norm(self, p='fro', dim=None, keepdim=False, out=None, dtype=None):
    """Compute the norm value of elements along the given dimension.

    Parameters
    ----------
    p : {'fro', 1, 2}, optional
        The norm order.
    dim : Union[int, Sequence[int]], optional
        The dimension to reduce.
    keepdim : bool, optional, default=False
        Keep the reduced dimension or not.
    dtype : str, optional
        The data type to cast to.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.norm(...)`_

    """
    return math_ops.norm(self, p, dim, keepdim, dtype=dtype)


def normal_(self, mean=0, std=1):
    r"""Fill self from a normal distribution.

    .. math:: \text{self} \sim \mathcal{N}(\mu, \sigma^{2})

    Parameters
    ----------
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    size = self.size()
    return Function.apply(
        'RandomNormal', self.device, [], outputs=[self],
        dtype=self.dtype, mean=float(mean), std=float(std),
        ndim=len(size), dims=size)


def permute(self, *dims):
    """Return a tensor with the new order of dimensions.

    Parameters
    ----------
    dims : int...
        The output dimensions.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_ops.permute(self, nest.flatten(dims))


def permute_(self, *dims):
    """Reorder the dimensions.

    Parameters
    ----------
    dims : Union[Sequence[int], int...]
        The new order of dimensions.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_ops.permute(self, nest.flatten(dims), self)


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
    `torch.pow(...)`_

    """
    return math_ops.pow(self, exponent)


def reciprocal(self):
    r"""Compute the reciprocal.

    .. math:: \text{out} = \frac{1}{\text{self}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.reciprocal(...)`_

    """
    return math_ops.reciprocal(self)


def reciprocal_(self):
    r"""Compute the reciprocal.

    .. math:: \text{self} = \frac{1}{\text{self}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.reciprocal(...)`_

    """
    return math_ops.reciprocal(self, self)


def repeat(self, *sizes):
    """Repeat elements along each dimension.

    Parameters
    ----------
    sizes : int...
        The repetition for each dimension.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return array_ops.tile(self, nest.flatten(sizes))


def reshape(self, *shape):
    """Return a tensor with the same data but a different shape.

    Parameters
    ----------
    shape : Union[Sequence[int], int...]
        The new shape.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.reshape(...)`_

    """
    return array_ops.reshape(self, nest.flatten(shape))


def reshape_(self, *shape):
    """Change into a new shape with the same data.

    Parameters
    ----------
    shape : Union[Sequence[int], int...]
        The new shape.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.reshape(...)`_

    """
    return array_ops.reshape(self, nest.flatten(shape), self)


def roll(self, shifts, dims=None):
    """Return a tensor of rolled elements.

    Parameters
    ----------
    shifts : Union[int, Sequence[int]]
        The rolling offset of each dimension.
    dims : Union[int, Sequence[int]], optional
        The dimension to roll.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.roll(...)`_

    """
    return array_ops.roll(self, shifts, dims)


def round(self):
    r"""Return a tensor taken the round of elements.

    .. math:: \text{out} = \lfloor \text{self} \rceil

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.round(...)`_

    """
    return math_ops.round(self)


def round_(self):
    r"""Set to the round of elements.

    .. math:: \text{self} = \lfloor \text{self} \rceil

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.round(...)`_

    """
    return math_ops.round(self, self)


def rsqrt(self):
    r"""Compute the reciprocal square root.

    .. math:: \text{out} = \frac{1}{\sqrt{\text{self}}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.rsqrt(...)`_

    """
    return math_ops.rsqrt(self)


def rsqrt_(self):
    r"""Compute the reciprocal square root.

    .. math:: \text{self} = \frac{1}{\sqrt{\text{self}}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.rsqrt(...)`_

    """
    return math_ops.rsqrt(self, self)


def scatter(self, dim, index, src):
    """Return a tensor with elements updated from the source.

    Parameters
    ----------
    dim : int
        The dimension of index values.
    index : dragon.vm.torch.Tensor
        The index tensor.
    src : Union[dragon.vm.torch.Tensor, number]
        The tensor to update from.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.scatter(...)`_

    """
    return array_ops.scatter(self, dim, index, src)


def scatter_(self, dim, index, src, reduce=None):
    """Update elements from the source.

    Parameters
    ----------
    dim : int
        The dimension of index values.
    index : dragon.vm.torch.Tensor
        The index tensor.
    src : Union[dragon.vm.torch.Tensor, number]
        The tensor to update from.
    reduce : str, optional
        ``'add'`` or ``'multiply'``.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.scatter(...)`_

    """
    if reduce:
        if reduce == 'add':
            return self.scatter_add_(dim, index, src)
        elif reduce == 'multiply':
            to_mul = constant_ops.ones_like(self, self.dtype, device=self.device)
            with grad_mode.no_grad():
                to_mul.scatter_(dim, index, src)
            return math_ops.mul(self, to_mul, self)
        else:
            raise ValueError('Unknown reduction: ' + reduce)
    return array_ops.scatter(self, dim, index, src, self)


def scatter_add(self, dim, index, src):
    """Return a tensor with elements added from the source.

    Parameters
    ----------
    dim : int
        The dimension of index values.
    index : dragon.vm.torch.Tensor
        The index tensor.
    src : Union[dragon.vm.torch.Tensor, number]
        The tensor to add from.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.scatter_add(...)`_

    """
    return array_ops.scatter_add(self, dim, index, src)


def scatter_add_(self, dim, index, src):
    """Add elements from the source.

    Parameters
    ----------
    dim : int
        The dimension of index values.
    index : dragon.vm.torch.Tensor
        The index tensor.
    src : Union[dragon.vm.torch.Tensor, number]
        The tensor to add from.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.scatter_add(...)`_

    """
    return array_ops.scatter_add(self, dim, index, src, self)


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
        _, value = constant_ops.remove_scalars(self, value)
        starts, sizes = _process_index(key)
        return Function.apply(
            'Assign', self.device, [self, value], outputs=[self],
            ndim=len(starts), starts=starts, sizes=sizes)


def sign(self):
    r"""Return a tensor taken the sign indication of elements.

    .. math::
        \text{out}_{i} =
            \begin{cases}
                -1, & \text{ if } \text{self}_{i} < 0 \\
                 0, & \text{ if } \text{self}_{i} = 0 \\
                 1, & \text{ if } \text{self}_{i} > 0
            \end{cases}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.sign(...)`_

    """
    return math_ops.sign(self)


def sign_(self):
    r"""Set to the sign indication of elements.

    .. math::
        \text{self}_{i} =
            \begin{cases}
                -1, & \text{ if } \text{self}_{i} < 0 \\
                 0, & \text{ if } \text{self}_{i} = 0 \\
                 1, & \text{ if } \text{self}_{i} > 0
            \end{cases}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.sign(...)`_

    """
    return math_ops.sign(self, self)


def sin(self):
    r"""Compute the sin.

    .. math:: \text{out} = \sin(\text{self})

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.sin(...)`_

    """
    return math_ops.sin(self)


def sort(self, dim=-1, descending=False):
    """Return the sorted elements.

    Parameters
    ----------
    dim : int, optional, default=-1
        The dimension to sort elements.
    descending : bool, optional, default=False
        Sort in the descending order or not.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The value and index tensor.

    See Also
    --------
    `torch.sort(...)`_

    """
    return sort_ops.sort(self, dim, descending)


def split(self, split_size_or_sections, dim=0, copy=True):
    """Return the split chunks along the given dimension.

    Parameters
    ----------
    split_size_or_sections : Union[int, Sequence[int]
        The number or size of chunks.
    dim : int, optional, default=0
        The dimension to split.
    copy : bool, optional, default=True
        Copy or create the views of input.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The output tensors.

    See Also
    --------
    `torch.split(...)`_

    """
    return array_ops.split(self, split_size_or_sections, dim, copy)


def sqrt(self):
    r"""Compute the square root.

    .. math:: \text{out} = \sqrt{\text{self}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.sqrt(...)`_

    """
    return math_ops.sqrt(self)


def sqrt_(self):
    r"""Compute the square root.

    .. math:: \text{self} = \sqrt{\text{self}}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.sqrt(...)`_

    """
    return math_ops.sqrt(self, self)


def square(self):
    r"""Compute the square of input.

    .. math:: \text{out} = \text{self}^{2}

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.square(...)`_

    """
    return math_ops.square(self)


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

    See Also
    --------
    `torch.squeeze(...)`_

    """
    return array_ops.squeeze(self, dim)


def squeeze_(self, dim=None):
    """Remove the dimensions with size 1.

    Parameters
    ----------
    dim : Union[int, Sequence[int]], optional
        The dimension(s) to remove.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.squeeze(...)`_

    """
    return array_ops.squeeze(self, dim, self)


def sum(self, dim=None, keepdim=False):
    """Compute the sum value of elements along the given dimension.

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

    See Also
    --------
    `torch.sum(...)`_

    """
    return math_ops.sum(self, dim, keepdim)


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
    `torch.sub(...)`_

    """
    return math_ops.sub(self, other)


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
        The output tensor.

    See Also
    --------
    `torch.sub(...)`_

    """
    return math_ops.sub(self, other, self)


def topk(self, k, dim=-1, largest=True, sorted=True):
    """Return the top k-largest or smallest elements.

    Parameters
    ----------
    k : int
        The number of top elements to select.
    dim : int, optional, default=-1
        The dimension to select elements.
    largest : bool, optional, default=True
        Return largest or smallest elements.
    sorted : bool, optional, default=True
        Whether to return elements in the sorted order.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The value and index tensor.

    See Also
    --------
    `torch.topk(...)`_

    """
    return sort_ops.topk(self, k, dim, largest, sorted)


def transpose(self, dim0, dim1):
    """Return a tensor with two dimensions swapped.

    Parameters
    ----------
    dim0 : int
        The first dimension to be transposed.
    dim1 : int
        The second dimension to be transposed.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.transpose(...)`_

    """
    return array_ops.transpose(self, dim0, dim1)


def transpose_(self, dim0, dim1):
    """Swap two dimensions.

    Parameters
    ----------
    dim0 : int
        The first dimension to be transposed.
    dim1 : int
        The second dimension to be transposed.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.transpose(...)`_

    """
    return array_ops.transpose(self, dim0, dim1, self)


def tril(self, k=0):
    r"""Return the lower triangular part.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                0, & \text{ if } j > i + k \\
                \text{self}_{ij}, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    k : int, optional, default=0
        Diagonal above which to zero elements.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.tril(...)`_

    """
    return array_ops.tril(self, k)


def tril_(self, k=0):
    r"""Set to the lower triangular part.

    .. math::
        \text{self}_{ij} =
            \begin{cases}
                0, & \text{ if } j > i + k \\
                \text{self}_{ij}, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    k : int, optional, default=0
        Diagonal above which to zero elements.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.tril(...)`_

    """
    return array_ops.tril(self, k, self)


def triu(self, k=0):
    r"""Return the upper triangular part.

    .. math::
        \text{out}_{ij} =
            \begin{cases}
                0, & \text{ if } j < i + k \\
                \text{self}_{ij}, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    k : int, optional, default=0
        Diagonal below which to zero elements.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.triu(...)`_

    """
    return array_ops.triu(self, k)


def triu_(self, k=0):
    r"""Set to the upper triangular part.

    .. math::
        \text{self}_{ij} =
            \begin{cases}
                0, & \text{ if } j < i + k \\
                \text{self}_{ij}, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    k : int, optional, default=0
        Diagonal below which to zero elements.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.triu(...)`_

    """
    return array_ops.triu(self, k, self)


def _type(self, dtype=None):
    """Return the data type.

    If ``dtype`` is not ``None``, converts to a new tensor.

    Parameters
    ----------
    dtype : str, optional
        The data type to convert to.

    Returns
    -------
    Union[str, dragon.vm.torch.Tensor]
        The data type or new tensor.

    """
    if dtype is None:
        return self.dtype
    return math_ops.cast(self, dtype)


def unbind(self, dim=0, copy=True):
    """Unpack to chunks along the given dimension.

    Parameters
    ----------
    dim : int, optional, default=0
        The dimension to unpack.
    copy : bool, optional, default=True
        Copy or create the views of input.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The output tensors.

    See Also
    --------
    `torch.unbind(...)`_

    """
    return array_ops.unbind(self, dim, copy)


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
        The output tensor.

    """
    size = self.size()
    return Function.apply(
        'RandomUniform', self.device, [], outputs=[self],
        dtype=self.dtype, low=float(low), high=float(high),
        ndim=len(size), dims=size)


def unique(self, return_inverse=False, return_counts=False, **kwargs):
    """Return the unique elements.

    Parameters
    ----------
    return_inverse : bool, optional, default=False
        Return the inverse index or not.
    return_counts : bool, optional, default=False
        Return the counts or not.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.
    dragon.vm.torch.Tensor, optional
        The inverse index tensor.
    dragon.vm.torch.Tensor, optional
        The counting tensor.

    See Also
    --------
    `torch.unique(...)`_

    """
    return array_ops.unique(self, return_inverse, return_counts, **kwargs)


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

    See Also
    --------
    `torch.unsqueeze(...)`_

    """
    return array_ops.unsqueeze(self, dim)


def unsqueeze_(self, dim):
    """Insert the dimensions of size 1.

    Parameters
    ----------
    dim : Union[int, Sequence[int]]
        The dimensions(s) to insert.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.unsqueeze(...)`_

    """
    return array_ops.unsqueeze(self, dim, self)


def where(self, condition, y):
    r"""Select the elements from two branches under the condition.

    .. math::
        \text{out}_{i} =
            \begin{cases}
                \text{self}_{i} & \text{ if } \text{condition}_{i} \\
                y_{i}, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    condition : dragon.vm.torch.Tensor
        The condition tensor.
    y : dragon.vm.torch.Tensor
        The tensor :math:`y`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.where(...)`_

    """
    return array_ops.where(condition, self, y)


def var(self, dim=None, keepdim=False):
    """Compute the variance value of elements along the given dimension.

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

    See Also
    --------
    `torch.var(...)`_

    """
    return math_ops.var(self, dim, keepdim)


def _process_index(item):
    """Process and normalize the index."""
    if not isinstance(item, (slice, tuple)):
        if not isinstance(item, int):
            raise ValueError('The index should be a integer.')
        item = (item,)
    if not isinstance(item, tuple):
        item = tuple([item])
    starts, sizes = [], []
    for i, ele in enumerate(item):
        if isinstance(ele, slice):
            if ele.start is None:
                starts.append(0)
            else:
                starts.append(ele.start)
            if ele.stop is None:
                sizes.append(-1)
            else:
                sizes.append(ele.stop - starts[-1])
                if sizes[-1] == 0:
                    raise ValueError(
                        'The starts and ends of dim {} can not be equal'
                        ', got {}:{}.'.format(i, starts[-1], ele.stop))
            if ele.step is not None:
                raise NotImplementedError
        elif isinstance(ele, int):
            starts.append(ele)
            sizes.append(0)
        else:
            raise TypeError('Unsupported index type: {}'.format(type(ele)))
    return starts, sizes


# Aliases
Tensor.abs = abs
Tensor.add = add
Tensor.add_ = add_
Tensor.addmm = addmm
Tensor.argmax = argmax
Tensor.argmin = argmin
Tensor.argsort = argsort
Tensor.atan2 = atan2
Tensor.backward = backward
Tensor.baddbmm = baddbmm
Tensor.baddbmm_ = baddbmm_
Tensor.bitwise_and = bitwise_and
Tensor.bitwise_and_ = bitwise_and_
Tensor.bitwise_not = bitwise_not
Tensor.bitwise_not_ = bitwise_not_
Tensor.bitwise_or = bitwise_or
Tensor.bitwise_or_ = bitwise_or_
Tensor.bitwise_xor = bitwise_xor
Tensor.bitwise_xor_ = bitwise_xor_
Tensor.bmm = bmm
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
Tensor.exp_ = exp_
Tensor.expand = expand
Tensor.fill_ = fill_
Tensor.flatten = flatten
Tensor.flatten_ = flatten_
Tensor.flip = flip
Tensor.fliplr = fliplr
Tensor.flipud = flipud
Tensor.float = _float
Tensor.float_ = _float_
Tensor.floor = floor
Tensor.floor_ = floor_
Tensor.gather = gather
Tensor.ge = ge
Tensor.gt = gt
Tensor.half = half
Tensor.half_ = half_
Tensor.index_select = index_select
Tensor.int = _int
Tensor.int_ = _int_
Tensor.isfinite = isfinite
Tensor.isinf = isinf
Tensor.isnan = isnan
Tensor.le = le
Tensor.long = long
Tensor.long_ = long_
Tensor.log = log
Tensor.log_ = log_
Tensor.logical_and = logical_and
Tensor.logical_not = logical_not
Tensor.logical_or = logical_or
Tensor.logical_xor = logical_xor
Tensor.logsumexp = logsumexp
Tensor.lt = lt
Tensor.masked_fill = masked_fill
Tensor.masked_fill_ = masked_fill_
Tensor.masked_select = masked_select
Tensor.matmul = matmul
Tensor.max = max
Tensor.maximum = maximum
Tensor.mean = mean
Tensor.min = min
Tensor.minimum = minimum
Tensor.mm = mm
Tensor.mul = mul
Tensor.mul_ = mul_
Tensor.multinomial = multinomial
Tensor.narrow = narrow
Tensor.ne = ne
Tensor.neg = neg
Tensor.neg_ = neg_
Tensor.new_empty = new_empty
Tensor.new_full = new_full
Tensor.new_tensor = new_tensor
Tensor.nonzero = nonzero
Tensor.norm = norm
Tensor.normal_ = normal_
Tensor.permute = permute
Tensor.permute_ = permute_
Tensor.pow = pow
Tensor.reciprocal = reciprocal
Tensor.reciprocal_ = reciprocal_
Tensor.repeat = repeat
Tensor.reshape = reshape
Tensor.reshape_ = reshape_
Tensor.roll = roll
Tensor.round = round
Tensor.round_ = round_
Tensor.rsqrt = rsqrt
Tensor.rsqrt_ = rsqrt_
Tensor.scatter = scatter
Tensor.scatter_ = scatter_
Tensor.scatter_add = scatter_add
Tensor.scatter_add_ = scatter_add_
Tensor.sign = sign
Tensor.sign_ = sign_
Tensor.sin = sin
Tensor.sort = sort
Tensor.sqrt = sqrt
Tensor.sqrt_ = sqrt_
Tensor.split = split
Tensor.square = square
Tensor.squeeze = squeeze
Tensor.squeeze_ = squeeze_
Tensor.sum = sum
Tensor.sub = sub
Tensor.sub_ = sub_
Tensor.topk = topk
Tensor.transpose = transpose
Tensor.transpose_ = transpose_
Tensor.tril = tril
Tensor.tril_ = tril_
Tensor.triu = triu
Tensor.triu_ = triu_
Tensor.type = _type
Tensor.unbind = unbind
Tensor.uniform_ = uniform_
Tensor.unique = unique
Tensor.unsqueeze = unsqueeze
Tensor.unsqueeze_ = unsqueeze_
Tensor.where = where
Tensor.var = var
Tensor.__getitem__ = getitem
Tensor.__radd__ = lambda self, value: math_ops._binary_func(value, self, 'Add')
Tensor.__rmul__ = lambda self, value: math_ops._binary_func(value, self, 'Mul')
Tensor.__rsub__ = lambda self, value: math_ops._binary_func(value, self, 'Sub')
Tensor.__rtruediv__ = lambda self, value: math_ops._binary_func(value, self, 'Div')
Tensor.__setitem__ = setitem
