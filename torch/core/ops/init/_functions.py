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
"""Init functions library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.autograd import function


class _Initializer(function.Function):
    """Base initializer function."""

    def __init__(self, key, dev, **kwargs):
        super(_Initializer, self).__init__(key, dev, **kwargs)
        self.dtype = kwargs.get('dtype', 'float32')

    def setup(self, ws, handle, shape):
        self.feed_arg(ws, '%s/dims' % handle, shape, 'int64')

    def forward(self, out, shape, shape_like=None):
        return self.dispatch(
            [] if shape_like is None else [shape_like], [out],
            callback=lambda ws, handle:
                self.setup(ws, handle, shape),
            no_grad=True,
        )


class Eye(_Initializer):
    """Eye function."""

    def __init__(self, key, dev, **kwargs):
        super(Eye, self).__init__(key, dev, **kwargs)
        self.k = kwargs.get('k', 0)

    def attributes(self):
        return {
            'op_type': 'Eye',
            'arguments': {
                'k': self.k,
                'dtype': self.dtype,
                'dims_desc': '${HANDLE}/dims',
            },
        }


class Fill(_Initializer):
    """Fill function."""

    def __init__(self, key, dev, **kwargs):
        super(Fill, self).__init__(key, dev, **kwargs)
        self.value = kwargs.get('value', 0.)

    def attributes(self):
        return {
            'op_type': 'Fill',
            'arguments': {
                'dtype': self.dtype,
                'value': float(self.value),
                'dims_desc': '${HANDLE}/dims',
            },
        }


class LinSpace(function.Function):
    """LinSpace function."""

    def __init__(self, key, dev, **kwargs):
        super(LinSpace, self).__init__(key, dev, **kwargs)
        self.num_intervals = kwargs.get('num_intervals', 1)
        self.dtype = kwargs.get('dtype', 'int64')
        self.axis = kwargs.get('axis', 0)

    def attributes(self):
        return {
            'op_type': 'LinSpace',
            'arguments': {
                'axis': self.axis,
                'dtype': self.dtype,
                'dims_desc': '${HANDLE}/dims',
                'start_desc': '${HANDLE}/start',
                'stop_desc': '${HANDLE}/stop',
            }
        }

    def setup(self, ws, handle, shape, starts, stops):
        self.feed_arg(ws, '%s/dims' % handle, shape, 'int64')
        self.feed_arg(ws, '%s/start' % handle, starts, 'float64')
        self.feed_arg(ws, '%s/stop' % handle, stops, 'float64')

    def forward(self, shape, starts, stops, out=None):
        return self.dispatch(
            [], [self.alloc(out)],
            callback=lambda ws, handle:
                self.setup(ws, handle, shape, starts, stops),
            no_grad=True,
        )


class Permutation(function.Function):
    """Permutation function."""

    def __init__(self, key, dev, **kwargs):
        super(Permutation, self).__init__(key, dev, **kwargs)
        self.dtype = kwargs.get('dtype', 'int64')

    def attributes(self):
        return {
            'op_type': 'Permutation',
            'arguments': {
                'dtype': self.dtype,
                'limit_desc': '${HANDLE}/limit',
            }
        }

    def setup(self, ws, handle, limit):
        self.feed_arg(ws, '{}/limit'.format(handle), limit, 'int64')

    def forward(self, limit, out=None):
        return self.dispatch(
            [], [self.alloc(out)],
            callback=lambda ws, handle:
                self.setup(ws, handle, limit),
            no_grad=True,
        )


class RandomNormal(_Initializer):
    """RandomNormal function."""

    def __init__(self, key, dev, **kwargs):
        super(RandomNormal, self).__init__(key, dev, **kwargs)
        self.mean = kwargs.get('mean', 0.)
        self.std = kwargs.get('std', 1.)

    def attributes(self):
        return {
            'op_type': 'RandomNormal',
            'arguments': {
                'dtype': self.dtype,
                'mean': float(self.mean),
                'std': float(self.std),
                'dims_desc': '${HANDLE}/dims',
            },
        }


class RandomUniform(_Initializer):
    """RandomUniform function."""

    def __init__(self, key, dev, **kwargs):
        super(RandomUniform, self).__init__(key, dev, **kwargs)
        self.low = kwargs.get('low', 0.)
        self.high = kwargs.get('high', 1.)

    def attributes(self):
        return {
            'op_type': 'RandomUniform',
            'arguments': {
                'dtype': self.dtype,
                'low': float(self.low),
                'high': float(self.high),
                'dims_desc': '${HANDLE}/dims',
            },
        }


class Range(function.Function):
    """Range function."""

    def __init__(self, key, dev, **kwargs):
        super(Range, self).__init__(key, dev, **kwargs)
        self.dtype = kwargs.get('dtype', 'int64')

    def attributes(self):
        return {
            'op_type': 'Range',
            'arguments': {
                'dtype': self.dtype,
                'slice_desc': '${HANDLE}/slice',
            }
        }

    def setup(self, ws, handle, slice_args):
        self.feed_arg(ws, '%s/slice' % handle, slice_args, 'float64')

    def forward(self, slice_args, out=None):
        return self.dispatch(
            [], [self.alloc(out)],
            callback=lambda ws, handle:
                self.setup(ws, handle, slice_args),
            no_grad=True,
        )
