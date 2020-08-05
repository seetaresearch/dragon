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
"""Bind tensor methods that executed eagerly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.eager import context
from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import ops
from dragon.core.ops import array_ops
from dragon.core.ops import array_ops_lib
from dragon.core.ops import control_flow_ops_lib
from dragon.core.ops import init_ops_lib
from dragon.core.ops import math_ops_lib


def add(self, other):
    """Compute the element-wise addition.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to add.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    """
    return _binary_op(self, other, 'Add')


def astype(self, dtype, inplace=False):
    """Cast the data type to a specific one.

    Parameters
    ----------
    dtype : str
        The specific data type.
    inplace : bool, optional, default=False
        Whether to do the cast in-place.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.cast(...)`_

    """
    return array_ops_lib.Cast \
        .instantiate(dtype=dtype).apply([self], inplace)


def constant(self, value=0):
    r"""Fill self with a scalar value.

    .. math:: \text{self} \leftarrow \text{value}

    Parameters
    ----------
    value : number, optional, default=0
        The value to fill.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.fill(...)`_

    """
    shape = self.shape
    return init_ops_lib.Fill \
        .instantiate(
            ndim=len(shape),
            value=value,
            dtype=self.dtype,
        ).apply(shape, out=self)


def copy(self):
    """Return a tensor with containing data copied.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.copy(...)`_

    """
    return control_flow_ops_lib.Copy \
        .instantiate().apply([self], None)


def div(self, other):
    """Compute the element-wise division.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to divide.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.div(...)`_

    """
    return _binary_op(self, other, 'Div')


def ge(self, other):
    """Compute element-wise greater-equal comparison.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to compare.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.greater_equal(...)`_

    """
    return _binary_op(self, other, 'GreaterEqual')


def getitem(self, item):
    """Select elements at the specific index.

    Parameters
    ----------
    item : Union[slice, int, dragon.EagerTensor]
        The index.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    """
    if isinstance(item, EagerTensor):
        return _masked_select(self, item)
    else:
        starts, sizes = _process_index(item)
        return _section_select(self, starts, sizes)


def glorot_normal(self, mode='fan_in', scale=2.0):
    r"""Fill self from a glorot normal distribution.

    .. math:: \text{self} \sim \mathcal{N}(0, \frac{scale}{\text{fan}})

    Parameters
    ----------
    mode : {'fan_in, 'fan_out', 'fan_avg'}, optional
        The mode to compute fans.
    scale : float, optional, default=2.0
        The scale factor of distribution.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.random.glorot_normal(...)`_

    """
    shape = self.shape
    return init_ops_lib.GlorotNormal \
        .instantiate(
            ndim=len(shape),
            dtype=self.dtype,
            mode=mode.lower(),
            scale=float(scale),
        ).apply(shape, out=self)


def glorot_uniform(self, mode='fan_in', scale=3.0):
    r"""Fill self from a glorot uniform distribution.

    .. math:: \text{self} \sim \mathcal{U}(-\sqrt{\frac{scale}{\text{fan}}},
                                            \sqrt{\frac{scale}{\text{fan}}})

    Parameters
    ----------
    mode : {'fan_in, 'fan_out', 'fan_avg'}, optional
        The mode to compute fans.
    scale : float, optional, default=3.0
        The scale factor of distribution.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.random.glorot_uniform(...)`_

    """
    shape = self.shape
    return init_ops_lib.GlorotUniform \
        .instantiate(
            ndim=len(shape),
            dtype=self.dtype,
            mode=mode.lower(),
            scale=float(scale),
        ).apply(shape, out=self)


def gt(self, other):
    """Compute element-wise greater comparison.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to compare.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.greater(...)`_

    """
    return _binary_op(self, other, 'Greater')


def iadd(self, other):
    """Compute the element-wise addition.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to add.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.math.add(...)`_

    """
    return _binary_op(self, other, 'Add', [self])


def idiv(self, other):
    """Compute the element-wise division.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to divide.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.math.div(...)`_

    """
    return _binary_op(self, other, 'Div', [self])


def imul(self, other):
    """Compute the element-wise multiplication.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to multiply.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.math.mul(...)`_

    """
    return _binary_op(self, other, 'Mul', [self])


def isub(self, other):
    """Compute the element-wise subtraction.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to subtract.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.math.sub(...)`_

    """
    return _binary_op(self, other, 'Sub', [self])


def le(self, other):
    """Compute element-wise less-equal comparison.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to compare.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.less_equal(...)`_

    """
    return _binary_op(self, other, 'LessEqual')


def lt(self, other):
    """Compute element-wise less comparison.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to compare.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.less(...)`_

    """
    return _binary_op(self, other, 'Less')


def mul(self, other):
    """Compute the element-wise multiplication.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to multiply.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.mul(...)`_

    """
    return _binary_op(self, other, 'Mul')


def neg(self):
    """Compute the element-wise negative.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.negative(...)`_

    """
    return _unary_op(self, 'Neg')


def normal(self, mean=0, std=1):
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
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.random.normal(...)`_

    """
    shape = self.shape
    return init_ops_lib.RandomNormal \
        .instantiate(
            ndim=len(shape),
            dtype=self.dtype,
            mean=float(mean),
            std=float(std),
        ).apply(shape, out=self)


def radd(self, other):
    """Compute the element-wise addition.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to add.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.add(...)`_

    """
    return _binary_op(other, self, 'Add')


def rdiv(self, other):
    """Compute the element-wise division.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to be divided.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.div(...)`_

    """
    return _binary_op(other, self, 'Div')


def reshape(self, shape):
    """Return a tensor containing the same data with new shape.

    Parameters
    ----------
    shape : Sequence[int]
        The new shape.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.reshape(...)`_

    """
    with context.eager_mode():
        return array_ops.reshape(self, shape=shape)


def rmul(self, other):
    """Compute the element-wise multiplication.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to multiply.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.mul(...)`_

    """
    return _binary_op(other, self, 'Mul')


def rsub(self, other):
    """Compute the element-wise subtraction.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to be subtracted.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.sub(...)`_

    """
    return _binary_op(other, self, 'Sub')


def setitem(self, key, value):
    """Set elements at the specific index.

    Parameters
    ----------
    key : Union[slice, int, dragon.EagerTensor]
        The index.
    value : Union[dragon.EagerTensor, number]
        The value to set.

    """
    if isinstance(key, EagerTensor):
        _masked_assign(self, value, key)
    else:
        starts, sizes = _process_index(key)
        _section_assign(self, value, starts, sizes)


def sub(self, other):
    """Compute the element-wise subtraction.

    Parameters
    ----------
    other : Union[dragon.EagerTensor, number]
        The value to subtract.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.sub(...)`_

    """
    return _binary_op(self, other, 'Sub')


def truncated_normal(self, mean=0, std=1):
    r"""Fill self from a truncated normal distribution.

    .. math:: \text{self} \sim \mathcal{TN}(\mu, \sigma^{2}, \mu - 2\sigma, \mu + 2\sigma)

    Parameters
    ----------
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.random.truncated_normal(...)`_

    """
    shape = self.shape
    return init_ops_lib.TruncatedNormal \
        .instantiate(
            ndim=len(shape),
            dtype=self.dtype,
            mean=float(mean),
            std=float(std),
        ).apply(shape, out=self)


def uniform(self, low=0, high=1):
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
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.random.uniform(...)`_

    """
    shape = self.shape
    return init_ops_lib.RandomUniform \
        .instantiate(
            ndim=len(shape),
            dtype=self.dtype,
            low=float(low),
            high=float(high),
        ).apply(shape, out=self)


def _binary_op(a, b, op_type, outputs=(None,)):
    """Apply the general binary operation."""
    return math_ops_lib.BinaryOp \
        .instantiate(op_type=op_type) \
        .apply(ops.remove_binary_scalar([a, b]), outputs)


def _masked_assign(ref, value, mask):
    """Apply the mask-assign operation."""
    value = ops.scalar_to_tensor(value, ref.dtype)
    return control_flow_ops_lib.MaskedAssign \
        .instantiate().apply([ref, value, mask])


def _masked_select(x, mask):
    """Apply the mask-select operation."""
    return array_ops_lib.MaskedSelect \
        .instantiate().apply([x, mask])


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
                        'The starts and ends of axis {} '
                        'can not be equal, got {}:{}.'
                        .format(i, starts[-1], ele.stop))
            if ele.step is not None:
                raise NotImplementedError
        elif isinstance(ele, int):
            starts.append(ele)
            sizes.append(0)
        else:
            raise TypeError(
                'Unsupported type of index: {}'
                .format(type(ele)))
    return starts, sizes


def _section_assign(ref, value, starts, sizes):
    """Apply the section-assign operation."""
    value = ops.scalar_to_tensor(value, ref.dtype)
    return control_flow_ops_lib.Assign \
        .instantiate(ndim=len(starts) if starts is not None else 0) \
        .apply([ref, value], starts, sizes)


def _section_select(x, starts, sizes):
    """Apply the section-select operation."""
    return array_ops_lib.Slice \
        .instantiate(ndim=len(starts)).apply([x], starts, sizes)


def _unary_op(x, op_type):
    """Apply the general unary operation."""
    return math_ops_lib.UnaryOp \
        .instantiate(op_type=op_type).apply([x])


# Aliases
EagerTensor.astype = astype
EagerTensor.constant = constant
EagerTensor.copy = copy
EagerTensor.glorot_normal = glorot_normal
EagerTensor.glorot_uniform = glorot_uniform
EagerTensor.normal = normal
EagerTensor.reshape = reshape
EagerTensor.truncated_normal = truncated_normal
EagerTensor.uniform = uniform
EagerTensor.__add__ = add
EagerTensor.__ge__ = ge
EagerTensor.__getitem__ = getitem
EagerTensor.__gt__ = gt
EagerTensor.__iadd__ = iadd
EagerTensor.__imul__ = imul
EagerTensor.__isub__ = isub
EagerTensor.__itruediv__ = idiv
EagerTensor.__le__ = le
EagerTensor.__lt__ = lt
EagerTensor.__mul__ = mul
EagerTensor.__neg__ = neg
EagerTensor.__radd__ = radd
EagerTensor.__rmul__ = rmul
EagerTensor.__rsub__ = rsub
EagerTensor.__rtruediv__ = rdiv
EagerTensor.__setitem__ = setitem
EagerTensor.__sub__ = sub
EagerTensor.__truediv__ = div
