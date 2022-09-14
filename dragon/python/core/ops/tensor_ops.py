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

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.framework.tensor import Tensor
from dragon.core.ops import array_ops
from dragon.core.ops import constant_ops


def add(self, other):
    """Compute the element-wise addition.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to add.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.add(...)`_

    """
    return _apply_binary_op([self, other], 'Add')


def _and(self, other):
    """Compute the element-wise AND bitwise operation.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.bitwise.bitwise_and(...)`_

    """
    return _apply_binary_op([self, other], 'BitwiseAnd')


def astype(self, dtype, copy=True):
    """Convert the data type to a specific one.

    Parameters
    ----------
    dtype : str
        The specific data type.
    copy : bool, optional, default=True
        Return a new tensor or converted in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.cast(...)`_

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Cast', [self], outputs=[None] if copy else [self], dtype=dtype)
    return OpLib.add('Cast', [self], dtype=dtype)


def copy(self):
    """Return a tensor with containing data copied.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.identity(...)`_

    """
    return _apply_unary_op([self], 'Identity')


def div(self, other):
    """Compute the element-wise division.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to divide.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.div(...)`_

    """
    return _apply_binary_op([self, other], 'Div')


def eq(self, other):
    """Compute element-wise equal comparison.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.equal(...)`_

    """
    return _apply_binary_op([self, other], 'Equal')


def fill(self, value):
    r"""Fill self with a scalar value.

    .. math:: \text{self} \leftarrow \text{value}

    Parameters
    ----------
    value : number
        The value to fill.

    Returns
    -------
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.fill(...)`_

    """
    return _apply_init_op(self, 'Fill', value=float(value))


def ge(self, other):
    """Compute element-wise greater-equal comparison.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.greater_equal(...)`_

    """
    return _apply_binary_op([self, other], 'GreaterEqual')


def getitem(self, item):
    """Select elements at the specific index.

    Parameters
    ----------
    item : Union[slice, int, dragon.Tensor]
        The index.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    gather_args = []
    if isinstance(item, Tensor):
        if item.dtype == 'bool' or item.dtype == 'uint8':
            if context.executing_eagerly():
                return OpLib.execute('BooleanMask', [self, item])
            return OpLib.add('BooleanMask', [self, item])
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
        axis, index = gather_args[0]
        if context.executing_eagerly():
            return OpLib.execute(
                'Gather', [self, index], axis=axis, end_axis=None)
        return OpLib.add('Gather', [self, index], axis=axis)
    elif len(gather_args) > 1:
        raise NotImplementedError
    starts, sizes = _process_index(item)
    if context.executing_eagerly():
        return OpLib.execute(
            'Slice', [self], ndim=len(starts), starts=starts, sizes=sizes)
    return OpLib.add('Slice', [self], starts=starts, sizes=sizes)


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
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.random.glorot_normal(...)`_

    """
    return _apply_init_op(self, 'GlorotNormal',
                          mode=mode.lower(), scale=float(scale))


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
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.random.glorot_uniform(...)`_

    """
    return _apply_init_op(self, 'GlorotUniform',
                          mode=mode.lower(), scale=float(scale))


def gt(self, other):
    """Compute element-wise greater comparison.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.greater(...)`_

    """
    return _apply_binary_op([self, other], 'Greater')


def iadd(self, other):
    """Compute the element-wise addition.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to add.

    Returns
    -------
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.math.add(...)`_

    """
    return _apply_binary_op([self, other], 'Add', [self])


def iand(self, other):
    """Compute the element-wise AND bitwise operation.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.bitwise.bitwise_and(...)`_

    """
    return _apply_binary_op([self, other], 'BitwiseAnd', [self])


def idiv(self, other):
    """Compute the element-wise division.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to divide.

    Returns
    -------
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.math.div(...)`_

    """
    return _apply_binary_op([self, other], 'Div', [self])


def imul(self, other):
    """Compute the element-wise multiplication.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to multiply.

    Returns
    -------
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.math.mul(...)`_

    """
    return _apply_binary_op([self, other], 'Mul', [self])


def invert(self):
    """Compute the element-wise NOT bitwise operation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.bitwise.invert(...)`_

    """
    return _apply_unary_op([self], 'BitwiseNot')


def ior(self, other):
    """Compute the element-wise OR bitwise operation.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.bitwise.bitwise_or(...)`_

    """
    return _apply_binary_op([self, other], 'BitwiseOr', [self])


def isub(self, other):
    """Compute the element-wise subtraction.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to subtract.

    Returns
    -------
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.math.sub(...)`_

    """
    return _apply_binary_op([self, other], 'Sub', [self])


def ixor(self, other):
    """Compute the element-wise XOR bitwise operation.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.bitwise.bitwise_xor(...)`_

    """
    return _apply_binary_op([self, other], 'BitwiseXor', [self])


def le(self, other):
    """Compute element-wise less-equal comparison.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.less_equal(...)`_

    """
    return _apply_binary_op([self, other], 'LessEqual')


def lt(self, other):
    """Compute element-wise less comparison.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.less(...)`_

    """
    return _apply_binary_op([self, other], 'Less')


def matmul(self, other):
    """Compute the matrix multiplication.

    Parameters
    ----------
    other : dragon.Tensor
        The value to multiply.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.matmul(...)`_

    """
    return _apply_binary_op([self, other], 'MatMul')


def mul(self, other):
    """Compute the element-wise multiplication.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to multiply.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.mul(...)`_

    """
    return _apply_binary_op([self, other], 'Mul')


def ne(self, other):
    """Compute element-wise not-equal comparison.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compare.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.not_equal(...)`_

    """
    return _apply_binary_op([self, other], 'NotEqual')


def neg(self):
    """Compute the element-wise negative.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.negative(...)`_

    """
    return _apply_unary_op([self], 'Neg')


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
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.random.normal(...)`_

    """
    return _apply_init_op(self, 'RandomNormal',
                          mean=float(mean), std=float(std))


def _or(self, other):
    """Compute the element-wise OR bitwise operation.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.bitwise.bitwise_or(...)`_

    """
    return _apply_binary_op([self, other], 'BitwiseOr')


def radd(self, other):
    """Compute the element-wise addition.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to add.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.add(...)`_

    """
    return _apply_binary_op([other, self], 'Add')


def rand(self, other):
    """Compute the element-wise AND bitwise operation.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.bitwise.bitwise_and(...)`_

    """
    return _apply_binary_op([other, self], 'BitwiseAnd')


def rdiv(self, other):
    """Compute the element-wise division.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to be divided.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.div(...)`_

    """
    return _apply_binary_op([other, self], 'Div')


def reshape(self, shape, copy=True):
    """Return a tensor containing the same data with new shape.

    Parameters
    ----------
    shape : Union[Sequence[int], dragon.Tensor]
        The output shape.
    copy : bool, optional, default=True
        Return a new tensor or reshape in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.reshape(...)`_

    """
    return array_ops.reshape(self, shape=shape, copy=copy)


def rmul(self, other):
    """Compute the element-wise multiplication.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to multiply.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.mul(...)`_

    """
    return _apply_binary_op([other, self], 'Mul')


def ror(self, other):
    """Compute the element-wise OR bitwise operation.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.bitwise.bitwise_or(...)`_

    """
    return _apply_binary_op([other, self], 'BitwiseOr')


def rsub(self, other):
    """Compute the element-wise subtraction.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to be subtracted.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.sub(...)`_

    """
    return _apply_binary_op([other, self], 'Sub')


def rxor(self, other):
    """Compute the element-wise XOR bitwise operation.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.bitwise.bitwise_xor(...)`_

    """
    return _apply_binary_op([other, self], 'BitwiseXor')


def setitem(self, key, value):
    """Set elements at the specific index.

    Parameters
    ----------
    key : Union[slice, int, dragon.Tensor]
        The index.
    value : Union[dragon.Tensor, number]
        The value to set.

    """
    if context.executing_eagerly():
        value = constant_ops.scalar(value, self.dtype)
        if isinstance(key, Tensor):
            return OpLib.execute('Where', [key, value, self], outputs=[self])
        else:
            starts, sizes = _process_index(key)
            return OpLib.execute(
                'Assign', [self, value], outputs=[self],
                ndim=len(starts) if starts is not None else 0,
                starts=starts, sizes=sizes)
    else:
        if isinstance(key, Tensor):
            raise RuntimeError(
                'Assigning via mask is an ambiguous behavior in graph mode. '
                'Use `dragon.where(...)` instead.')
        else:
            raise RuntimeError(
                'Assigning via slices is an ambiguous behavior in graph mode. '
                'Use `dragon.assign(...)` instead.')


def sub(self, other):
    """Compute the element-wise subtraction.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to subtract.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.math.sub(...)`_

    """
    return _apply_binary_op([self, other], 'Sub')


def transpose(self, perm=None, copy=True):
    """Return a tensor with permuted axes.

    Parameters
    ----------
    perm : Union[Sequence[int], dragon.Tensor]], optional
        The output permutation.
    copy : bool, optional, default=True
        Return a new tensor or transpose in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.transpose(...)`_

    """
    return array_ops.transpose(self, perm=perm, copy=copy)


def truncated_normal(self, mean=0, std=1):
    r"""Fill self from a truncated normal distribution.

    .. math:: \text{self} \sim \mathcal{TN}(\mu, \sigma^{2},
                                            \mu - 2\sigma, \mu + 2\sigma)

    Parameters
    ----------
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.

    Returns
    -------
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.random.truncated_normal(...)`_

    """
    return _apply_init_op(
        self, 'TruncatedNormal', mean=float(mean), std=float(std))


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
    dragon.Tensor
        The self.

    See Also
    --------
    `dragon.random.uniform(...)`_

    """
    return _apply_init_op(
        self, 'RandomUniform', low=float(low), high=float(high))


def xor(self, other):
    """Compute the element-wise XOR bitwise operation.

    Parameters
    ----------
    other : Union[dragon.Tensor, number]
        The value to compute with.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.bitwise.bitwise_xor(...)`_

    """
    return _apply_binary_op([self, other], 'BitwiseXor')


def _apply_binary_op(inputs, op_type, outputs=(None,)):
    """Apply the binary operator."""
    inputs = constant_ops.remove_scalars(inputs)
    if context.executing_eagerly():
        return OpLib.execute(op_type, inputs, outputs=outputs)
    return OpLib.add(op_type, inputs)


def _apply_init_op(self, op_type, **kwargs):
    """Apply the initialization operator."""
    shape = self.shape
    if shape is None or None in shape:
        raise ValueError('Excepted the certain shape to initialize data.')
    return OpLib.execute(
        op_type, [], outputs=[self], dtype=self.dtype,
        ndim=len(shape), dims=shape, **kwargs)


def _apply_unary_op(inputs, op_type, outputs=(None,)):
    """Apply the unary operator."""
    if context.executing_eagerly():
        return OpLib.execute(op_type, inputs, outputs=outputs)
    return OpLib.add(op_type, inputs)


def _process_index(item):
    """Process and normalize the index."""
    if not isinstance(item, (slice, tuple)):
        if not isinstance(item, int):
            raise ValueError('The index should be a integer.')
        item = (item,)
    if not isinstance(item, tuple):
        item = tuple([item])
    starts, sizes = [], []
    for i, elem in enumerate(item):
        if isinstance(elem, slice):
            if elem.start is None:
                starts.append(0)
            else:
                starts.append(elem.start)
            if elem.stop is None:
                sizes.append(-1)
            else:
                sizes.append(elem.stop - starts[-1])
                if sizes[-1] == 0:
                    raise ValueError(
                        'The starts and ends of axis {} can not be equal'
                        ', got {}:{}.'.format(i, starts[-1], elem.stop))
            if elem.step is not None:
                raise NotImplementedError
        elif isinstance(elem, int):
            starts.append(elem)
            sizes.append(0)
        else:
            raise TypeError('Unsupported index type: {}'.format(type(elem)))
    return starts, sizes


# Aliases
Tensor.astype = astype
Tensor.copy = copy
Tensor.fill = fill
Tensor.glorot_normal = glorot_normal
Tensor.glorot_uniform = glorot_uniform
Tensor.normal = normal
Tensor.reshape = reshape
Tensor.transpose = transpose
Tensor.truncated_normal = truncated_normal
Tensor.uniform = uniform
Tensor.__add__ = add
Tensor.__and__ = _and
Tensor.__eq__ = eq
Tensor.__ge__ = ge
Tensor.__getitem__ = getitem
Tensor.__gt__ = gt
Tensor.__iadd__ = iadd
Tensor.__iand__ = iand
Tensor.__idiv__ = idiv
Tensor.__imul__ = imul
Tensor.__invert__ = invert
Tensor.__ior__ = ior
Tensor.__isub__ = isub
Tensor.__itruediv__ = idiv
Tensor.__ixor__ = ixor
Tensor.__le__ = le
Tensor.__lt__ = lt
Tensor.__matmul__ = matmul
Tensor.__mul__ = mul
Tensor.__ne__ = ne
Tensor.__neg__ = neg
Tensor.__or__ = _or
Tensor.__radd__ = radd
Tensor.__rand__ = rand
Tensor.__rmul__ = rmul
Tensor.__ror__ = ror
Tensor.__rtruediv__ = rdiv
Tensor.__rsub__ = rsub
Tensor.__rxor__ = rxor
Tensor.__setitem__ = setitem
Tensor.__sub__ = sub
Tensor.__truediv__ = div
Tensor.__xor__ = xor
