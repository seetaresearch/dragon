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
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(_Initializer, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)
        self.dtype = kwargs.get('dtype', 'float32')

    def feed(self, ws, handle, shape):
        """
        Feed the shape of the given shape.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            shape: (int): write your description
        """
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/dims[{}]'.format(handle, i),
                shape[i], 'int64')

    def forward(self, out, shape, shape_like=None):
        """
        Implementation of the layer.

        Args:
            self: (todo): write your description
            out: (array): write your description
            shape: (int): write your description
            shape_like: (todo): write your description
        """
        return self.dispatch(
            [] if shape_like is None else [shape_like], [out],
            callback=lambda ws, handle:
                self.feed(ws, handle, shape),
            no_grad=True,
        )


class Eye(_Initializer):
    """Eye function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Eye, self).__init__(key, dev, **kwargs)
        self.k = kwargs.get('k', 0)

    def attributes(self):
        """
        : return : class : dict

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Eye',
            'arguments': {
                'k': self.k,
                'dtype': self.dtype,
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'.format(n)
                    for n in range(self.ndim)],
            },
        }


class Fill(_Initializer):
    """Fill function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Fill, self).__init__(key, dev, **kwargs)
        self.value = kwargs.get('value', 0.)

    def attributes(self):
        """
        A dictionary of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Fill',
            'arguments': {
                'dtype': self.dtype,
                'value': float(self.value),
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'.format(n)
                    for n in range(self.ndim)],
            },
        }


class LinSpace(function.Function):
    """LinSpace function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(LinSpace, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)
        self.num_intervals = kwargs.get('num_intervals', 1)
        self.dtype = kwargs.get('dtype', 'int64')
        self.axis = kwargs.get('axis', 0)

    def attributes(self):
        """
        A dictionary of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'LinSpace',
            'arguments': {
                'axis': self.axis,
                'dtype': self.dtype,
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'.format(n)
                    for n in range(self.ndim)],
                'start_descs': [
                    '${{HANDLE}}/start[{}]'
                    .format(n) for n in range(self.num_intervals)],
                'stop_descs': [
                    '${{HANDLE}}/stop[{}]'
                    .format(n) for n in range(self.num_intervals)],
            }
        }

    def feed(self, ws, handle, shape, starts, stops):
        """
        Feed a subset of the given shape.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            shape: (int): write your description
            starts: (todo): write your description
            stops: (todo): write your description
        """
        for i, dim in enumerate(shape):
            self.feed_arg(
                ws, '{}/dims[{}]'.format(handle, i),
                dim, 'int64')
        for i in range(len(starts)):
            self.feed_arg(
                ws, '{}/start[{}]'.format(handle, i),
                starts[i], 'float64')
            self.feed_arg(
                ws, '{}/stop[{}]'.format(handle, i),
                stops[i], 'float64')

    def forward(self, shape, starts, stops, out=None):
        """
        Evaluates the given shape.

        Args:
            self: (todo): write your description
            shape: (int): write your description
            starts: (todo): write your description
            stops: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch(
            [], [self.alloc(out)],
            callback=lambda ws, handle:
                self.feed(ws, handle, shape, starts, stops),
            no_grad=True,
        )


class Permutation(function.Function):
    """Permutation function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Permutation, self).__init__(key, dev, **kwargs)
        self.dtype = kwargs.get('dtype', 'int64')

    def attributes(self):
        """
        A dict of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Permutation',
            'arguments': {
                'dtype': self.dtype,
                'limit_desc': '${HANDLE}/limit',
            }
        }

    def feed(self, ws, handle, limit):
        """
        Handle a message.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            limit: (int): write your description
        """
        self.feed_arg(ws, '{}/limit'.format(handle), limit, 'int64')

    def forward(self, limit, out=None):
        """
        Runs the queue.

        Args:
            self: (todo): write your description
            limit: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch(
            [], [self.alloc(out)],
            callback=lambda ws, handle:
                self.feed(ws, handle, limit),
            no_grad=True,
        )


class RandomNormal(_Initializer):
    """RandomNormal function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(RandomNormal, self).__init__(key, dev, **kwargs)
        self.mean = kwargs.get('mean', 0.)
        self.std = kwargs.get('std', 1.)

    def attributes(self):
        """
        Returns a dictionary of the attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'RandomNormal',
            'arguments': {
                'dtype': self.dtype,
                'mean': float(self.mean),
                'std': float(self.std),
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'.format(n)
                    for n in range(self.ndim)],
            },
        }


class RandomUniform(_Initializer):
    """RandomUniform function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(RandomUniform, self).__init__(key, dev, **kwargs)
        self.low = kwargs.get('low', 0.)
        self.high = kwargs.get('high', 1.)

    def attributes(self):
        """
        Return a dictionary of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'RandomUniform',
            'arguments': {
                'dtype': self.dtype,
                'low': float(self.low),
                'high': float(self.high),
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'.format(n)
                    for n in range(self.ndim)],
            },
        }


class Range(function.Function):
    """Range function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Range, self).__init__(key, dev, **kwargs)
        self.num_args = kwargs.get('num_args', 3)
        self.dtype = kwargs.get('dtype', 'int64')

    def attributes(self):
        """
        Return a dictionary of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Range',
            'arguments': {
                'dtype': self.dtype,
                'slice_descs': [
                    '${{HANDLE}}/slice[{}]'
                    .format(n) for n in range(self.num_args)],
            }
        }

    def feed(self, ws, handle, slice_args):
        """
        Feed a slice.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            slice_args: (dict): write your description
        """
        for i in range(len(slice_args)):
            self.feed_arg(
                ws, '{}/slice[{}]'.format(handle, i),
                slice_args[i], 'float64')

    def forward(self, slice_args, out=None):
        """
        Evaluate the layer.

        Args:
            self: (todo): write your description
            slice_args: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch(
            [], [self.alloc(out)],
            callback=lambda ws, handle:
            self.feed(ws, handle, slice_args),
            no_grad=True,
        )
