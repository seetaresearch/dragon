# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.execute_engine import RunOperator
from dragon.vm.torch.ops.factory import get_module
from dragon.vm.torch.autograd.grad_mode import no_grad
from dragon.vm.torch.ops.primitive import MakeContext
from dragon.vm.torch.ops.arithmetic import _fundamental, _rfundamental
from dragon.vm.torch.ops.control_flow import _copy
from dragon.vm.torch.ops.ndarray import \
    (reshape, _fill, _reduce, _arg_reduce,  _crop)
from dragon.vm.torch.ops.modules.dtype import AsType


##############################################
#                                            #
#                   BASE                     #
#                                            #
##############################################


def copy_(self, src):
    """Copy the elements from ``src`` into this tensor and return ``self``.

    Parameters
    ----------
    src : vm.torch.Tensor
        The source tensor.

    Returns
    -------
    vm.torch.Tensor
        The ``self`` tensor.

    """
    return _copy(self, src)


Tensor.copy_ = copy_


##############################################
#                                            #
#                INITIALIZER                 #
#                                            #
##############################################


def fill_(self, value):
    """Fill self tensor with the specified value.

    Parameters
    ----------
    value : numerical type

    Returns
    -------
    vm.torch.Tensor
        The self.

    """
    return _fill(self, self.shape, value)


def uniform_(self, low=0, high=1):
    """Fill self tensor with the specified uniform distribution.

    Parameters
    ----------
    low : numerical type
        The lower bound.
    high : numerical type
        The higher bound.

    Returns
    -------
    vm.torch.Tensor
        The self.

    """
    # TODO(PhyscalX): To support various dtypes, not only float32.
    arguments = {'low': float(low), 'high': float(high), 'dims': self.shape}
    inputs = []; outputs = [self]; ctx = MakeContext(inputs, outputs)
    meta = ('ONCE', 'RandomUniform', ctx)
    return RunOperator(inputs, outputs, meta, **arguments)


def normal_(self, mean=0, std=1):
    """Fill self tensor with the specified normal distribution.

    Parameters
    ----------
    mean : numerical type
        The mean(mu) of normal distribution.
    std : numerical type
        The std(sigma) of normal distribution.

    Returns
    -------
    vm.torch.Tensor
        The self.

    """
    # TODO(PhyscalX): To support various dtypes, not only float32.
    arguments = {'mean': float(mean), 'std': float(std), 'dims': self.shape}
    inputs = []; outputs = [self]; ctx = MakeContext(inputs, outputs)
    meta = ('ONCE', 'RandomNormal', ctx)
    return RunOperator(inputs, outputs, meta, **arguments)


Tensor.fill_ = fill_
Tensor.uniform_ = uniform_
Tensor.normal_ = normal_


##############################################
#                                            #
#                 ARITHMETIC                 #
#                                            #
##############################################


def add(self, value):
    """See ``torch.add()``

    Parameters
    ----------
    value : vm.torch.Tensor, int or float
        The value tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(self, value, op='Add')


def add_(self, value):
    """Inplace of ``torch.add()``

    Parameters
    ----------
    value : vm.torch.Tensor, int or float
        The value tensor.

    Returns
    -------
    vm.torch.Tensor
        The self.

    """
    return _fundamental(self, value, out=self, op='Add')


def radd(self, value):
    return _rfundamental(self, value, op='RAdd')


def sub(self, value):
    """Subtract the ``self`` and ``value`` into the output tensor.

    Parameters
    ----------
    value : vm.torch.Tensor, int or float
        The value tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(self, value, op='Sub')


def sub_(self, value):
    """Inplace of ``Tensor.sub()``

    Parameters
    ----------
    value : vm.torch.Tensor, int or float
        The value tensor.

    Returns
    -------
    vm.torch.Tensor
        The self.

    """
    return _fundamental(self, value, out=self, op='Sub')


def rsub(self, value):
    return _rfundamental(self, value, op='RSub')


def mul(self, value):
    """Multiply the ``self`` and ``value`` into the output tensor.

    Parameters
    ----------
    value : vm.torch.Tensor, int or float
        The value tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(self, value, op='Mul')


def mul_(self, value):
    """Inplace of ``Tensor.mul()``

    Parameters
    ----------
    value : vm.torch.Tensor, int or float
        The value tensor.

    Returns
    -------
    vm.torch.Tensor
        The self.

    """
    return _fundamental(self, value, out=self, op='Mul')


def rmul(self, value):
    return _rfundamental(self, value, op='RMul')


def div(self, value):
    """Divide the ``self`` and ``value`` into the output tensor.

    Parameters
    ----------
    value : vm.torch.Tensor, int or float
        The value tensor.

    Returns
    -------
    vm.torch.Tensor
        The output tensor.

    """
    return _fundamental(self, value, op='Div')


def div_(self, value):
    """Inplace of ``Tensor.div()``

    Parameters
    ----------
    value : vm.torch.Tensor, int or float
        The value tensor.

    Returns
    -------
    vm.torch.Tensor
        The self.

    """
    return _fundamental(self, value, out=self, op='Div')


def rdiv(self, value):
    return _rfundamental(self, value, op='RDiv')


Tensor.add = add
Tensor.add_ = add_
Tensor.__radd__ = radd
Tensor.sub = sub
Tensor.sub_ = sub_
Tensor.__rsub__ = rsub
Tensor.mul = mul
Tensor.mul_ = mul_
Tensor.__rmul__ = rmul
Tensor.div = div
Tensor.div_ = div_
Tensor.__rdiv__ = rdiv
Tensor.__rtruediv__ = rdiv


##############################################
#                                            #
#                   ARRAY                    #
#                                            #
##############################################


def view(self, *args):
    if self._static_shape:
        raise RuntimeError('Can not view a leaf variable, it owns the static sizes.')
    return reshape(self, shape=args)


def view_as(self, other):
    if not isinstance(other, Tensor):
        raise ValueError('The other should be a torch tensor.')
    if self._static_shape:
        raise RuntimeError('Can not view a leaf variable, it owns the static sizes.')
    return reshape(self, shape=None, shape_like=other)


def crop(self, starts, ends):
    return _crop(self, starts, ends)


def mean(self, dim=None, keepdim=False):
    return _reduce(self, 'MEAN', dim, keepdim)


def sum(self, dim=None, keepdim=False):
    return _reduce(self, 'SUM', dim, keepdim)


def max(self, dim=None, keepdim=False):
    return _arg_reduce(self, 'MAX', dim, keepdim)


def min(self, dim=None, keepdim=False):
    return _arg_reduce(self, 'MIN', dim, keepdim)


Tensor.view = view
Tensor.view_as = view_as
Tensor.mean = mean
Tensor.sum = sum
Tensor.max = max
Tensor.min = min
Tensor._crop = crop


##############################################
#                                            #
#                    TYPE                    #
#                                            #
##############################################


def _type_to(input, dtype='float32', inplace=False):
    if dtype == input._dtype: return input
    ctx = MakeContext(inputs=[input])
    key = 'torch/ops/astype/{}:{}/dtype:{}/inplace:{}'.format(
        ctx[0].lower(), ctx[1], dtype, 'true' if inplace else 'false')
    module = get_module(AsType, key, ctx, dtype=dtype, inplace=inplace)
    with no_grad():
        return module.forward(input)


def _type(self, dtype=None):
    """Return the data type of this tensor.

    If ``dtype`` is not ``None``, cast ``self`` to the new tensor.

    Parameters
    ----------
    dtype : str
        The specified type.

    Returns
    -------
    str or vm.torch.Tensor
        The data type or the new tensor.

    """
    if dtype is None:
        return 'torch.' + self._type2str()
    else:
        return _type_to(self, dtype=dtype)


Tensor.type = _type
Tensor.half = lambda self: _type_to(self, dtype='float16', inplace=False)
Tensor.half_ = lambda self: _type_to(self, dtype='float16', inplace=True)
Tensor.float = lambda self: _type_to(self, dtype='float32', inplace=False)
Tensor.float_ = lambda self: _type_to(self, dtype='float32', inplace=True)
Tensor.double = lambda self: _type_to(self, dtype='float64', inplace=False)
Tensor.double_ = lambda self: _type_to(self, dtype='float64', inplace=True)
Tensor.byte = lambda self: _type_to(self, dtype='uint8', inplace=False)
Tensor.byte_ = lambda self: _type_to(self, dtype='uint8', inplace=True)
Tensor.int = lambda self: _type_to(self, dtype='int32', inplace=False)
Tensor.int_ = lambda self: _type_to(self, dtype='int32', inplace=True)
Tensor.long = lambda self: _type_to(self, dtype='int64', inplace=False)
Tensor.long_ = lambda self: _type_to(self, dtype='int64', inplace=True)