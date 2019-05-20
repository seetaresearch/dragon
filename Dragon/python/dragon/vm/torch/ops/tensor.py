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
from dragon.vm.torch.ops.factory import get_module
from dragon.vm.torch.ops.primitive import MakeDevice
from dragon.vm.torch.ops.modules.array import Cast

from dragon.vm.torch.ops.builtin import (
    _fill, _uniform, _normal, multinomial,
    _fundamental, _rfundamental,
    log, exp, sqrt, clamp,
    _reshape, squeeze, unsqueeze,
    _permute, _repeat, narrow, _index,
    _assign, _masked_assign,
    index_select, masked_select,
    mean, sum, max, min,
    gt, lt, eq, ne, ge, le,
    where, nonzero,
)


def _type_to(input, dtype='float32', inplace=False):
    if dtype == input.dtype: return input
    dev = MakeDevice(inputs=[input])
    key = 'Cast/{}/dtype:{}/inplace:{}'.format(
        dev, dtype, 'true' if inplace else 'false')
    module = get_module(Cast, key, dev, dtype=dtype, inplace=inplace)
    return module.forward(input)


Tensor.fill_ = lambda self, value: _fill(self, self.shape, value)
Tensor.masked_fill_ = lambda *args, **kwargs: _masked_assign(*args, **kwargs)
Tensor.uniform_ = lambda self, low=0, high=1: _uniform(self, self.shape, low, high)
Tensor.normal_ = lambda self, mean=0, std=1: _normal(self, self.shape, mean, std)
Tensor.multinomial = lambda *args, **kwargs: multinomial(*args, **kwargs)


Tensor.add = lambda self, value: _fundamental(self, value, 'Add')
Tensor.add_ = lambda self, value: _fundamental(self, value, 'Add', self)
Tensor.__radd__ = lambda self, value: _rfundamental(self, value, 'RAdd')
Tensor.sub = lambda self, value: _fundamental(self, value, 'Sub')
Tensor.sub_ = lambda self, value: _fundamental(self, value, 'Sub', self)
Tensor.__rsub__ = lambda self, value: _rfundamental(self, value, 'RSub')
Tensor.mul = lambda self, value: _fundamental(self, value, 'Mul')
Tensor.mul_ = lambda self, value: _fundamental(self, value, 'Mul', self)
Tensor.__rmul__ = lambda self, value: _rfundamental(self, value, 'RMul')
Tensor.div = lambda self, value: _fundamental(self, value, 'Div')
Tensor.div_ = lambda self, value: _fundamental(self, value, 'Div', self)
Tensor.__rdiv__ = lambda self, value: _rfundamental(self, value, 'RDiv')
Tensor.__rtruediv__ = lambda self, value: _rfundamental(self, value, 'RDiv')
Tensor.clamp = lambda *args, **kwargs: clamp(*args, **kwargs)
Tensor.clamp_ = lambda self, min=None, max=None: clamp(self, min, max, self)
Tensor.log = lambda *args, **kwargs: log(*args, **kwargs)
Tensor.exp = lambda *args, **kwargs: exp(*args, **kwargs)
Tensor.sqrt = lambda *args, **kwargs: sqrt(*args, **kwargs)


Tensor.squeeze = lambda *args, **kwargs: squeeze(*args, **kwargs)
Tensor.squeeze_ = lambda self, dim: squeeze(self, dim, self)
Tensor.unsqueeze = lambda *args, **kwargs: unsqueeze(*args, **kwargs)
Tensor.unsqueeze_ = lambda self, dim: unsqueeze(self, dim, self)
Tensor.view = lambda self, *shape: _reshape(self, shape)
Tensor.view_as = lambda *args, **kwargs: _reshape(*args, **kwargs)
Tensor.permute = lambda self, *dims: _permute(self, dims)
Tensor.repeat = lambda self, *args: _repeat(self, args)
Tensor.mean = lambda *args, **kwargs: mean(*args, **kwargs)
Tensor.sum = lambda *args, **kwargs: sum(*args, **kwargs)
Tensor.max = lambda *args, **kwargs: max(*args, **kwargs)
Tensor.min = lambda *args, **kwargs: min(*args, **kwargs)
Tensor.gt = lambda *args, **kwargs: gt(*args, **kwargs)
Tensor.ge = lambda *args, **kwargs: ge(*args, **kwargs)
Tensor.lt = lambda *args, **kwargs: lt(*args, **kwargs)
Tensor.le = lambda *args, **kwargs: le(*args, **kwargs)
Tensor.eq = lambda *args, **kwargs: eq(*args, **kwargs)
Tensor.ne = lambda *args, **kwargs: ne(*args, **kwargs)
Tensor.nonzero = lambda *args, **kwargs: nonzero(*args, **kwargs)
Tensor.where = lambda self, condition, y: where(condition, self, y)
Tensor.narrow = lambda *args, **kwargs: narrow(*args, **kwargs)
Tensor._index = lambda *args, **kwargs: _index(*args, **kwargs)
Tensor._assign = lambda *args, **kwargs: _assign(*args, **kwargs)
Tensor.index_select = lambda *args, **kwargs: index_select(*args, **kwargs)
Tensor.masked_select = lambda *args, **kwargs: masked_select(*args, **kwargs)


Tensor.half = lambda self: _type_to(self, dtype='float16', inplace=False)
Tensor.half_ = lambda self: _type_to(self, dtype='float16', inplace=True)
Tensor.float = lambda self: _type_to(self, dtype='float32', inplace=False)
Tensor.float_ = lambda self: _type_to(self, dtype='float32', inplace=True)
Tensor.double = lambda self: _type_to(self, dtype='float64', inplace=False)
Tensor.double_ = lambda self: _type_to(self, dtype='float64', inplace=True)
Tensor.byte = lambda self: _type_to(self, dtype='uint8', inplace=False)
Tensor.byte_ = lambda self: _type_to(self, dtype='uint8', inplace=True)
Tensor.char = lambda self: _type_to(self, dtype='int8', inplace=False)
Tensor.char_ = lambda self: _type_to(self, dtype='int8', inplace=True)
Tensor.int = lambda self: _type_to(self, dtype='int32', inplace=False)
Tensor.int_ = lambda self: _type_to(self, dtype='int32', inplace=True)
Tensor.long = lambda self: _type_to(self, dtype='int64', inplace=False)
Tensor.long_ = lambda self: _type_to(self, dtype='int64', inplace=True)
Tensor.type = lambda self, dtype = None: _type_to(self, dtype=dtype) \
    if dtype is not None else 'torch.' + self._type2str()