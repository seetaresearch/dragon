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

"""Bind tensor methods executed symbolically."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph.op_def import OpDef
from dragon.core.autograph.tensor import Tensor
from dragon.core.eager import context
from dragon.core.framework import ops
from dragon.core.framework import workspace
from dragon.core.ops import array_ops


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
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.cast(...)`_ : Cast the data type of input.

    """
    if self.dtype == dtype:
        return self
    inputs, outputs = ([], [self]) if inplace else ([self], None)
    return OpDef.apply('Cast', inputs, outputs, dtype=dtype)


def copy(self):
    """Return a tensor with containing data copied.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.copy(...)`_ : Copy the value to ref.

    """
    outputs = [Tensor(shape=self.shape, dtype=self.dtype)]
    return OpDef.apply('Copy', [self], [outputs])


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

    """
    return _binary_op(self, other, 'Div')


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

    """
    return _binary_op(self, other, 'GreaterEqual')


def getitem(self, item):
    """Select elements at the specific index.

    Parameters
    ----------
    item : Union[int, slice, dragon.Tensor]
        The index.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if isinstance(item, Tensor):
        return _masked_select(self, item)
    else:
        starts, sizes = _process_indices(item)
        return _section_select(self, starts, sizes)


def get_value(self):
    """Return the value of implementation.

    Returns
    -------
    numpy.ndarray
        The deep-copied value.

    """
    return workspace.get_workspace().fetch_tensor(self)


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

    """
    return _binary_op(self, other, 'Greater')


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

    """
    return _binary_op(self, other, 'LessEqual')


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

    """
    return _binary_op(self, other, 'Less')


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

    """
    return _binary_op(self, other, 'Mul')


def neg(self):
    """Compute the element-wise negative.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return _unary_op(self, 'Neg')


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

    """
    return _binary_op(other, self, 'Add')


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
    dragon.Tensor
        The output tensor.

    See Also
    --------
    `dragon.reshape(...)`_ : Change the dimensions of input.

    """
    with context.graph_mode():
        return array_ops.reshape(self, shape=shape)


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

    """
    return _binary_op(other, self, 'Mul')


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

    """
    return _binary_op(other, self, 'Sub')


def setitem(self, key, value):
    """Set elements at the specific index.

    Parameters
    ----------
    key : Union[int, slice, dragon.Tensor]
        The index.
    value : Union[dragon.Tensor, number]
        The value to set.

    """
    if isinstance(key, Tensor):
        _masked_assign(self, value, key)
    else:
        starts, sizes = _process_indices(key)
        _section_assign(self, value, starts, sizes)


def set_value(self, value):
    """Set value to the implementation.

    Parameters
    ----------
    value : array_like
        The value to set.

    Returns
    -------
    dragon.Tensor
        The self.

    """
    workspace.get_workspace().feed_tensor(self, value)
    return self


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

    """
    return _binary_op(self, other, 'Sub')


def _binary_op(a, b, op_type):
    """Create the general binary operator."""
    a, b = ops.remove_binary_scalar([a, b])
    return OpDef.apply(op_type, [a, b])


def _masked_assign(ref, value, mask):
    """Create the mask-assign operator."""
    value = ops.scalar_to_tensor(value, ref.dtype)
    return OpDef.apply('MaskedAssign', [value, mask], [ref])


def _masked_select(x, mask):
    """Create the mask-select operator."""
    return OpDef.apply('MaskedSelect', [x, mask])


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
    """Create the section-assign operator."""
    value = ops.scalar_to_tensor(value, ref.dtype)
    return OpDef.apply('Assign', [value], [ref], starts=starts, sizes=sizes)


def _section_select(x, starts, sizes):
    """Create the section-select operator."""
    return OpDef.apply('Slice', [x], starts=starts, sizes=sizes)


def _unary_op(x, op_type):
    """Create the general unary operator."""
    return OpDef.apply(op_type, [x])


# Aliases
Tensor.astype = astype
Tensor.copy = copy
Tensor.get_value = get_value
Tensor.reshape = reshape
Tensor.set_value = set_value
Tensor.__add__ = add
Tensor.__ge__ = ge
Tensor.__getitem__ = getitem
Tensor.__gt__ = gt
Tensor.__le__ = le
Tensor.__lt__ = lt
Tensor.__mul__ = mul
Tensor.__neg__ = neg
Tensor.__radd__ = radd
Tensor.__rmul__ = rmul
Tensor.__rtruediv__ = rdiv
Tensor.__rsub__ = rsub
Tensor.__setitem__ = setitem
Tensor.__sub__ = sub
Tensor.__truediv__ = div
