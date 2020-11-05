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
"""Array ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import device_spec
from dragon.core.framework.ops import Operator


class ArgReduce(Operator):
    """ArgReduce operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(ArgReduce, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', 'ArgMax')
        self.axis = kwargs.get('axis', None)
        self.keep_dims = kwargs.get('keep_dims', True)

    def attributes(self):
        """
        Returns the attributes for this axis

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': self.op_type,
            'arguments': {
                'axis': self.axis,
                'keep_dims': self.keep_dims,
            }
        }

    def forward(self, inputs):
        """
        Run the module inputs.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class Cast(Operator):
    """Cast operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Cast, self).__init__(key, dev, **kwargs)
        self.dtype = kwargs.get('dtype', 'float32')

    def attributes(self):
        """
        Return a dict.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Cast',
            'arguments': {'dtype': self.dtype},
        }

    def forward(self, inputs, inplace=False):
        """
        R forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            inplace: (bool): write your description
        """
        if inputs[0].dtype == self.dtype:
            return inputs[0]
        if inplace:
            return self.dispatch([], [self.alloc(inputs[0])], no_grad=True)
        return self.dispatch(inputs, [self.alloc()])


class ChannelAffine(Operator):
    """ChannelAffine operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the channel.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(ChannelAffine, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)
        self.num_axes = kwargs.get('num_axes', 1)

    def attributes(self):
        """
        : return : class attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'ChannelAffine',
            'arguments': {
                'axis': self.axis,
                'num_axes': self.num_axes,
            }
        }

    def forward(self, inputs, inplace=False):
        """
        Forward computation

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            inplace: (bool): write your description
        """
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class ChannelNormalize(Operator):
    """ChannelNormalize operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize channel.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(ChannelNormalize, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)
        self.ndim = kwargs.get('ndim', 0)
        self.mean = kwargs.get('mean', None)
        self.std = kwargs.get('std', None)
        self.dtype = kwargs.get('dtype', 'float32')

    def attributes(self):
        """
        Returns the attributes of the dataset

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'ChannelNormalize',
            'arguments': {
                'axis': self.axis,
                'mean': self.mean,
                'std': self.std,
                'dtype': self.dtype,
                'perm_descs': [
                    '${{HANDLE}}/perm[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, perm):
        """
        Feed a feed.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            perm: (todo): write your description
        """
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/perm[{}]'.format(handle, i),
                perm[i], 'int64')

    def forward(self, inputs, perm):
        """
        Parameters ---------- inputs : list of inputs.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            perm: (todo): write your description
        """
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, perm),
        )


class ChannelShuffle(Operator):
    """ChannelShuffle operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize channel channel.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(ChannelShuffle, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.group = kwargs.get('group', 1)

    def attributes(self):
        """
        A dictionary of the group attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'ChannelShuffle',
            'arguments': {
                'axis': self.axis,
                'group': self.group,
            }
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class Concat(Operator):
    """Concat operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Concat, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)

    def attributes(self):
        """
        Return the attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Concat',
            'arguments': {'axis': self.axis},
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class Cumulative(Operator):
    """Cumulative operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Cumulative, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.exclusive = kwargs.get('exclusive', False)
        self.reverse = kwargs.get('reverse', False)
        self.operation = kwargs.get('operation', 'Sum')

    def attributes(self):
        """
        The attributes of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Cum' + self.operation,
            'arguments': {
                'axis': self.axis,
                'exclusive': self.exclusive,
                'reverse': self.reverse,
            }
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class Expand(Operator):
    """Expand operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Expand, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        """
        A dictionary of the attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Expand',
            'arguments': {
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, dims):
        """
        Feed the given dimensions.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            dims: (int): write your description
        """
        for i, dim in enumerate(dims):
            self.feed_arg(
                ws, '{}/dims[{}]'.format(handle, i),
                dim, 'int64')

    def forward(self, inputs, dims):
        """
        Parameters ---------- inputs : ndarray. tensor ) dims ]

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            dims: (todo): write your description
        """
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, dims),
        )


class ExpandDims(Operator):
    """ExpandDims operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the underlying device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(ExpandDims, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', [0])

    def attributes(self):
        """
        Return the axes attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'ExpandDims',
            'arguments': {
                'axes': self.axes,
            }
        }

    def forward(self, inputs, inplace=False):
        """
        Forward computation

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            inplace: (bool): write your description
        """
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Flatten(Operator):
    """Flatten operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Flatten, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.num_axes = kwargs.get('num_axes', -1)
        self.keep_axes = kwargs.get('keep_axes', None)

    def attributes(self):
        """
        Return the axis attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Flatten',
            'arguments': {
                'axis': self.axis,
                'num_axes': self.num_axes,
                'keep_axes': self.keep_axes,
            }
        }

    def forward(self, inputs, inplace=False):
        """
        Forward computation

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            inplace: (bool): write your description
        """
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class IndexSelect(Operator):
    """IndexSelect operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(IndexSelect, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.num_axes = kwargs.get('num_axes', 1)

    def attributes(self):
        """
        : return : class attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'IndexSelect',
            'arguments': {
                'axis': self.axis,
                'num_axes': self.num_axes,
            }
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class LinSpace(Operator):
    """LinSpace operator."""

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

    def forward(self, shape, starts, stops, trainable=False):
        """
        See : meth : class.

        Args:
            self: (todo): write your description
            shape: (int): write your description
            starts: (todo): write your description
            stops: (todo): write your description
            trainable: (bool): write your description
        """
        out = self.dispatch(
            [], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, shape, starts, stops),
            no_grad=True,
        )
        out._requires_grad = trainable
        return out


class MaskedSelect(Operator):
    """MaskedSelect operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(MaskedSelect, self).__init__(key, dev, **kwargs)

    def attributes(self):
        """
        A dict of attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': 'MaskedSelect', 'arguments': {}}

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class Moments(Operator):
    """Moments operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the devices.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Moments, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)
        self.keep_dims = kwargs.get('keep_dims', True)

    def attributes(self):
        """
        The axes attributes for this axis.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Moments',
            'arguments': {
                'axes': self.axes,
                'keep_dims': self.keep_dims,
            }
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc(), self.alloc()])


class Multinomial(Operator):
    """Multinomial operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Multinomial, self).__init__(key, dev, **kwargs)
        self.epsilon = kwargs.get('epsilon', 0.)
        self.normalize = kwargs.get('normalize', False)
        self.num_samples = kwargs.get('num_samples', 1)

    def attributes(self):
        """
        Returns the number of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Multinomial',
            'arguments': {
                'epsilon': self.epsilon,
                'normalize': self.normalize,
                'num_samples': self.num_samples,
            }
        }

    def forward(self, inputs):
        """
        Run the module inputs.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class NonZero(Operator):
    """NonZero operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(NonZero, self).__init__(key, dev, **kwargs)

    def attributes(self):
        """
        A dict of attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': 'NonZero', 'arguments': {}}

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class OneHot(Operator):
    """OneHot operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize on on device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(OneHot, self).__init__(key, dev, **kwargs)
        self.depth = kwargs.get('depth', 1)
        self.on_value = kwargs.get('on_value', 1)
        self.off_value = kwargs.get('off_value', 0)

    def attributes(self):
        """
        Return a dict of all attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'OneHot',
            'arguments': {
                'depth': self.depth,
                'on_value': self.on_value,
                'off_value': self.off_value,
            }
        }

    def forward(self, inputs):
        """
        Run the module inputs.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class Pad(Operator):
    """Pad operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Pad, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)
        self.value = kwargs.get('value', 0.)
        self.mode = kwargs.get('mode', 'CONSTANT')

    def attributes(self):
        """
        Returns a dictionary of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Pad',
            'arguments': {
                'mode': self.mode,
                'value': self.value,
                'pads_descs': [
                    '${{HANDLE}}/pads[{}]'
                    .format(n) for n in range(self.ndim * 2)],
            }
        }

    def feed(self, ws, handle, pads):
        """
        Feed the given packet.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            pads: (int): write your description
        """
        for i, e in enumerate(pads):
            self.feed_arg(
                ws, '{}/pads[{}]'.format(handle, i),
                e, 'int64')

    def forward(self, inputs, pads):
        """
        Runs the layer.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            pads: (todo): write your description
        """
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, pads),
        )


class Permutation(Operator):
    """Permutation operator."""

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

    def forward(self, limit, trainable=False):
        """
        The forward forward forward.

        Args:
            self: (todo): write your description
            limit: (todo): write your description
            trainable: (bool): write your description
        """
        out = self.dispatch(
            [], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, limit),
            no_grad=True,
        )
        out._requires_grad = trainable
        return out


class Range(Operator):
    """Range operator."""

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

    def forward(self, slice_args, trainable=False):
        """
        Perform forward forward pass.

        Args:
            self: (todo): write your description
            slice_args: (todo): write your description
            trainable: (bool): write your description
        """
        out = self.dispatch(
            [], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, slice_args),
            no_grad=True,
        )
        out._requires_grad = trainable
        return out


class Reduce(Operator):
    """Reduce operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the operation.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Reduce, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)
        self.keep_dims = kwargs.get('keep_dims', True)
        self.operation = kwargs.get('operation', 'Sum')

    def attributes(self):
        """
        Return the attributes of the axes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Reduce' + self.operation,
            'arguments': {
                'axes': self.axes,
                'keep_dims': self.keep_dims,
            }
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class Repeat(Operator):
    """Repeat operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Repeat, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 2147483647)
        self.repeats = kwargs.get('repeats', 1)

    def attributes(self):
        """
        : return : class attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Repeat',
            'arguments': {
                'axis': self.axis,
                'repeats': self.repeats,
            }
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class Reshape(Operator):
    """Reshape operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Reshape, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        """
        A dictionary of the attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Reshape',
            'arguments': {
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, shape):
        """
        Feed the shape of the given shape.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            shape: (int): write your description
        """
        for i, e in enumerate(shape):
            self.feed_arg(
                ws, '{}/dims[{}]'.format(handle, i),
                e, 'int64')

    def forward(self, inputs, shape, inplace=False):
        """
        Parameters ---------- inputs : list )

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            shape: (int): write your description
            inplace: (bool): write your description
        """
        return self.dispatch(
            inputs, [self.alloc(inputs[0]) if inplace else self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, shape),
        )


class Slice(Operator):
    """Slice operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Slice, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        """
        A dictionary of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Slice',
            'arguments': {
                'starts_descs': [
                    '${{HANDLE}}/starts[{}]'
                    .format(n) for n in range(self.ndim)],
                'sizes_descs': [
                    '${{HANDLE}}/sizes[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, starts, sizes):
        """
        Add a new message.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            starts: (todo): write your description
            sizes: (int): write your description
        """
        for i in range(len(starts)):
            self.feed_arg(
                ws, '{}/starts[{}]'.format(handle, i),
                starts[i], 'int64')
            self.feed_arg(
                ws, '{}/sizes[{}]'.format(handle, i),
                sizes[i], 'int64')

    def forward(self, inputs, starts, sizes):
        """
        Parameters ---------- inputs : intended )

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            starts: (todo): write your description
            sizes: (int): write your description
        """
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, starts, sizes),
        )


class Shape(Operator):
    """Shape operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize a device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Shape, self).__init__(key, dev, **kwargs)
        self._device = device_spec.DeviceSpec()

    def attributes(self):
        """
        A dict of attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': 'Shape', 'arguments': {}}

    def forward(self, inputs):
        """
        Run the module inputs.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class Sort(Operator):
    """Sort operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Sort, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)
        self.descending = kwargs.get('descending', False)

    def attributes(self):
        """
        : return : class : axis

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Sort',
            'arguments': {
                'axis': self.axis,
                'descending': self.descending,
            }
        }

    def forward(self, inputs):
        """
        R forward forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc(), self.alloc()], no_grad=True)


class Split(Operator):
    """Split operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Split, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.size_splits = kwargs.get('size_splits', None)
        self.slice_points = kwargs.get('slice_points', None)

    def attributes(self):
        """
        The slice attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Split',
            'arguments': {
                'axis': self.axis,
                'size_splits': self.size_splits,
                'slice_points': self.slice_points,
            }
        }

    def forward(self, inputs, chunks):
        """
        Parameters ---------- inputs : iterable ) ] ).

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            chunks: (todo): write your description
        """
        outputs = [self.alloc() for _ in range(chunks)]
        return self.dispatch(inputs, outputs)


class Squeeze(Operator):
    """Squeeze operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Squeeze, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)

    def attributes(self):
        """
        Return the axes attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Squeeze',
            'arguments': {'axes': self.axes},
        }

    def forward(self, inputs, inplace=False):
        """
        Forward computation

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            inplace: (bool): write your description
        """
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Stack(Operator):
    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Stack, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)

    def attributes(self):
        """
        Return the attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Stack',
            'arguments': {'axis': self.axis},
        }

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])


class Tile(Operator):
    """Tile operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Tile, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        """
        A dictionary of the attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Tile',
            'arguments': {
                'repeats_descs': [
                    '${{HANDLE}}/repeats[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, repeats):
        """
        Feeds a feed.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            repeats: (int): write your description
        """
        for i, size in enumerate(repeats):
            self.feed_arg(
                ws, '{}/repeats[{}]'.format(handle, i),
                size, 'int64')

    def forward(self, inputs, repeats):
        """
        Runs the forward forward.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            repeats: (int): write your description
        """
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, repeats),
        )


class Transpose(Operator):
    """Transpose operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Transpose, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        """
        A dictionary of the attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Transpose',
            'arguments': {
                'perm_descs': [
                    '${{HANDLE}}/perm[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, perm):
        """
        Feed a feed.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            perm: (todo): write your description
        """
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/perm[{}]'.format(handle, i),
                perm[i], 'int64')

    def forward(self, inputs, perm):
        """
        Parameters ---------- inputs : list of inputs.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            perm: (todo): write your description
        """
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, perm),
        )


class TopK(Operator):
    """TopK operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize kwargs.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(TopK, self).__init__(key, dev, **kwargs)
        self.k = kwargs.get('k', 1)
        self.axis = kwargs.get('axis', -1)
        self.largest = kwargs.get('largest', True)
        self.sorted = kwargs.get('sorted', True)

    def attributes(self):
        """
        Returns a dictionary of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'TopK',
            'arguments': {
                'k': self.k,
                'axis': self.axis,
                'largest': self.largest,
                'sorted': self.sorted,
            }
        }

    def forward(self, inputs):
        """
        R forward forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc(), self.alloc()], no_grad=True)


class Unique(Operator):
    """Unique operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize counts.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Unique, self).__init__(key, dev, **kwargs)
        self.return_inverse = kwargs.get('return_inverse', False)
        self.return_counts = kwargs.get('return_counts', False)
        self.num_outputs = 1 + self.return_inverse + self.return_counts

    def attributes(self):
        """
        The number of - attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Unique',
            'arguments': {
                'return_inverse': self.return_inverse,
                'return_counts': self.return_counts,
            }
        }

    def forward(self, inputs):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        outputs = [self.alloc() for _ in range(self.num_outputs)]
        return self.dispatch(inputs, outputs)


class Where(Operator):
    """Where operator."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Where, self).__init__(key, dev, **kwargs)

    def attributes(self):
        """
        A dict of attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': 'Where', 'arguments': {}}

    def forward(self, inputs):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return self.dispatch(inputs, [self.alloc()])
