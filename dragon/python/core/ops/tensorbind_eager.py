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

"""Bind tensor methods executed eagerly."""

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


def add(self, value):
    r"""Compute the element-wise addition.

    .. math:: \text{out} = \text{self} + \text{value}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to add.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.add(...)`_ : Compute the element-wise addition.

    """
    return _binary_op(self, value, 'Add')


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
    `dragon.cast(...)`_ : Cast the data type of input.

    """
    return array_ops_lib.Cast \
        .instantiate(
            dtype=dtype,
        ).apply([self], inplace)


def constant(self, value=0):
    r"""Fill self with a constant value.

    .. math:: \text{self} \leftarrow \text{value}

    Parameters
    ----------
    value : number, optional, default=0
        The constant value.

    Returns
    -------
    dragon.EagerTensor
        The self.

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
    `dragon.copy(...)`_ : Copy the value to ref.

    """
    return control_flow_ops_lib.Copy \
        .instantiate().apply([self], None)


def div(self, value):
    r"""Compute the element-wise division.

    .. math:: \text{out} = \text{self} \div \text{value}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to divide.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.div(...)`_ : Compute the element-wise division.

    """
    return _binary_op(self, value, 'Div')


def ge(self, other):
    r"""Compute element-wise greater-equal comparison.

    .. math:: \text{out} = (\text{self} \geq \text{other})

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
    `dragon.math.greater_equal(...)`_ : Compute element-wise greater-equal comparison.

    """
    return _binary_op(self, other, 'GreaterEqual')


def getitem(self, item):
    """Select the elements at the specific indices.

    Parameters
    ----------
    item : Union[int, slice, dragon.EagerTensor]
        The indices.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.slice(...)`_ : Select the elements according to the given sections.

    See Also
    --------
    `dragon.masked_select(...)`_ : Select the elements where the given mask is 1.

    """
    if isinstance(item, EagerTensor):
        return _masked_select(self, item)
    else:
        starts, sizes = _process_indices(item)
        return _section_select(self, starts, sizes)


def glorot_normal(self, mode='FAN_IN', scale=2.):
    r"""Fill self from a glorot normal distribution.

    .. math:: \text{self} \leftarrow N(0, \sqrt{\frac{scale}{\text{FAN}}})

    Parameters
    ----------
    mode : {'FAN_IN, 'FAN_OUT', 'FAN_AVG'}, optional
        The mode to compute fans.
    scale : number, optional, default=2.
        The scale factor of distribution.

    Returns
    -------
    dragon.EagerTensor
        The self.

    """
    shape = self.shape
    return init_ops_lib.GlorotNormal \
        .instantiate(
            ndim=len(shape),
            dtype=self.dtype,
            mode=mode.lower(),
            scale=float(scale),
        ).apply(shape, out=self)


def glorot_uniform(self, mode='FAN_IN', scale=3.):
    r"""Fill self from a glorot uniform distribution.

    .. math:: \text{self} \leftarrow U(
                -\sqrt{\frac{scale}{\text{FAN}}},
                \sqrt{\frac{scale}{\text{FAN}}}
            )

    Parameters
    ----------
    mode : {'FAN_IN, 'FAN_OUT', 'FAN_AVG'}, optional
        The mode to compute fans.
    scale : number, optional, default=3.
        The scale factor of distribution.

    Returns
    -------
    dragon.EagerTensor
        The self.

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
    r"""Compute element-wise greater comparison.

    .. math:: \text{out} = (\text{self} > \text{other})

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
    `dragon.math.greater(...)`_ : Compute element-wise greater comparison.

    """
    return _binary_op(self, other, 'Greater')


def iadd(self, value):
    r"""Compute the element-wise addition.

    .. math:: \text{self} \mathrel{+}= \text{value}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to add.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.math.add(...)`_ : Compute the element-wise addition.

    """
    return _binary_op(self, value, 'Add', [self])


def idiv(self, value):
    r"""Compute the element-wise division.

    .. math:: \text{self} \mathrel{\div}= \text{value}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to divide.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.math.div(...)`_ : Compute the element-wise division.

    """
    return _binary_op(self, value, 'Div', [self])


def imul(self, value):
    r"""Compute the element-wise multiplication.

    .. math:: \text{self} \mathrel{\times}= \text{value}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to multiply.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.math.mul(...)`_ : Compute the element-wise multiplication.

    """
    return _binary_op(self, value, 'Mul', [self])


def isub(self, value):
    r"""Compute the element-wise division.

    .. math:: \text{self} \mathrel{-}= \text{value}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to subtract.

    Returns
    -------
    dragon.EagerTensor
        The self.

    See Also
    --------
    `dragon.math.sub(...)`_ : Compute the element-wise subtraction.

    """
    return _binary_op(self, value, 'Sub', [self])


def le(self, other):
    r"""Compute element-wise less-equal comparison.

    .. math:: \text{out} = (\text{self} \leq \text{other})

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
    `dragon.math.less_equal(...)`_ : Compute element-wise less-equal comparison.

    """
    return _binary_op(self, other, 'LessEqual')


def lt(self, other):
    r"""Compute element-wise less comparison.

    .. math:: \text{out} = (\text{self} < \text{other})

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
    `dragon.math.less(...)`_ : Compute element-wise less comparison.

    """
    return _binary_op(self, other, 'Less')


def mul(self, value):
    r"""Compute the element-wise multiplication.

    .. math:: \text{out} = \text{self} \times \text{value}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to multiply.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.mul(...)`_ : Compute the element-wise multiplication.

    """
    return _binary_op(self, value, 'Mul')


def neg(self):
    r"""Compute the element-wise negative.

    .. math:: y = -x

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.negative(...)`_ : Compute the element-wise negative.

    """
    return _unary_op(self, 'Neg')


def normal(self, mean=0, std=1):
    r"""Fill self from a normal distribution.

    .. math:: \text{self} \leftarrow N(\mu, \sigma)

    Parameters
    ----------
    mean : number, optional, default=0
        The value of :math:`\mu`.
    std : number, optional, default=1
        The value of :math:`\sigma`.

    Returns
    -------
    dragon.EagerTensor
        The self.

    """
    shape = self.shape
    return init_ops_lib.RandomNormal \
        .instantiate(
            ndim=len(shape),
            dtype=self.dtype,
            mean=float(mean),
            std=float(std),
        ).apply(shape, out=self)


def radd(self, value):
    r"""Compute the element-wise addition.

    .. math:: \text{out} = \text{value} + \text{self}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to add.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.add(...)`_ : Compute the element-wise addition.

    """
    return _binary_op(value, self, 'Add')


def rdiv(self, value):
    r"""Compute the element-wise division.

    .. math:: \text{out} = \text{value} \div \text{self}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to be divided.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.div(...)`_ : Compute the element-wise division.

    """
    return _binary_op(value, self, 'Div')


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
    `dragon.reshape(...)`_ : Change the dimensions of input.

    """
    with context.eager_mode():
        return array_ops.reshape(self, shape=shape)


def rmul(self, value):
    r"""Compute the element-wise multiplication.

    .. math:: \text{out} = \text{value} \times \text{self}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to multiply.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.mul(...)`_ : Compute the element-wise multiplication.

    """
    return _binary_op(value, self, 'Mul')


def rsub(self, value):
    r"""Compute the element-wise subtraction.

    .. math:: \text{out} = \text{value} - \text{self}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to be subtracted.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.sub(...)`_ : Compute the element-wise subtraction.

    """
    return _binary_op(value, self, 'Sub')


def setitem(self, key, value):
    """Set the value at the specific indices.

    Parameters
    ----------
    key : Union[int, slice, dragon.EagerTensor]
        The indices.
    value : number or dragon.EagerTensor
        The value.

    See Also
    --------
    `dragon.assign(...)`_ : Assign the value to ref.

    See Also
    --------
    `dragon.masked_assign(...)`_ : Assign the value to ref where mask is 1.

    """
    if isinstance(key, EagerTensor):
        _masked_assign(self, value, key)
    else:
        starts, sizes = _process_indices(key)
        _section_assign(self, value, starts, sizes)


def sub(self, value):
    r"""Compute the element-wise subtraction.

    .. math:: \text{out} = \text{self} - \text{value}

    Parameters
    ----------
    value : Union[dragon.EagerTensor, number]
        The value to subtract.

    Returns
    -------
    dragon.EagerTensor
        The output tensor.

    See Also
    --------
    `dragon.math.sub(...)`_ : Compute the element-wise subtraction.

    """
    return _binary_op(self, value, 'Sub')


def truncated_normal(self, mean=0, std=1):
    r"""Fill self from a truncated normal distribution.

    .. math:: \text{self} \leftarrow TN(\mu, \sigma, \mu - 2\sigma, \mu + 2\sigma)

    Parameters
    ----------
    mean : number, optional, default=0
        The value of :math:`\mu`.
    std : number, optional, default=1
        The value of :math:`\sigma`.

    Returns
    -------
    dragon.EagerTensor
        The self.

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

    .. math:: \text{self} \leftarrow U(\alpha, \beta)

    Parameters
    ----------
    low : number, optional, default=0
        The value of :math:`\alpha`.
    high : number, optional, default=1
        The value of :math:`\beta`.

    Returns
    -------
    dragon.EagerTensor
        The self.

    """
    shape = self.shape
    return init_ops_lib.RandomUniform \
        .instantiate(
            ndim=len(shape),
            dtype=self.dtype,
            low=float(low),
            high=float(high),
        ).apply(shape, out=self)


def _binary_op(a, b, op_type, outputs=None):
    """Apply the general binary operation."""
    return math_ops_lib.Binary \
        .instantiate(op_type=op_type) \
        .apply(ops.remove_binary_scalar([a, b]), outputs)


def _masked_assign(ref, value, mask):
    """Apply the mask-assign operation."""
    value = ops.scalar_to_tensor(value, ref.dtype)
    return control_flow_ops_lib.MaskedAssign \
        .instantiate().apply([ref, value], mask)


def _masked_select(x, mask):
    """Apply the mask-select operation."""
    return array_ops_lib.MaskedSelect \
        .instantiate().apply([x, mask])


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


def _section_assign(ref, value, starts, sizes):
    """Apply the section-assign operation."""
    value = ops.scalar_to_tensor(value, ref.dtype)
    return control_flow_ops_lib.Assign \
        .instantiate(
            ndim=len(starts) if starts is not None else 0,
        ).apply([ref, value], starts, sizes)


def _section_select(x, starts, sizes):
    """Apply the section-select operation."""
    return array_ops_lib.Slice \
        .instantiate(
            ndim=len(starts),
        ).apply([x], starts, sizes)


def _unary_op(x, op_type):
    """Apply the general unary operation."""
    return math_ops_lib.Unary.instantiate(op_type=op_type).apply(x)


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
EagerTensor.__div__ = div
EagerTensor.__ge__ = ge
EagerTensor.__getitem__ = getitem
EagerTensor.__gt__ = gt
EagerTensor.__iadd__ = iadd
EagerTensor.__idiv__ = idiv
EagerTensor.__imul__ = imul
EagerTensor.__isub__ = isub
EagerTensor.__le__ = le
EagerTensor.__lt__ = lt
EagerTensor.__mul__ = mul
EagerTensor.__neg__ = neg
EagerTensor.__rdiv__ = rdiv
EagerTensor.__rmul__ = rmul
EagerTensor.__rtruediv__ = rdiv
EagerTensor.__rsub__ = rsub
EagerTensor.__setitem__ = setitem
EagerTensor.__sub__ = sub
