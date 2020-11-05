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
"""Array functions library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.autograd import function


class ArgReduce(function.Function):
    """ArgReduce function."""

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
        Return the attributes for this axis

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': self.op_type,
            'arguments': {
                'axis': self.axis,
                'keep_dims': self.keep_dims,
            },
        }

    def forward(self, input, out=None):
        """
        R forward forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input], [self.alloc(out)], no_grad=True)


class Assign(function.Function):
    """Assign function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(Assign, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        """
        A dictionary of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Assign',
            'arguments': {
                'starts_descs': [
                    '${{HANDLE}}/starts[{}]'
                    .format(n) for n in range(self.ndim)],
                'sizes_descs': [
                    '${{HANDLE}}/sizes[{}]'
                    .format(n) for n in range(self.ndim)],
            },
        }

    def feed(self, ws, handle, starts, sizes):
        """
        Add the arguments.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            starts: (todo): write your description
            sizes: (int): write your description
        """
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/starts[{}]'.format(handle, i),
                starts[i], 'int64')
            self.feed_arg(
                ws, '{}/sizes[{}]'.format(handle, i),
                sizes[i], 'int64')

    def forward(self, out, starts, sizes, input):
        """
        Implement forward computation.

        Args:
            self: (todo): write your description
            out: (array): write your description
            starts: (todo): write your description
            sizes: (int): write your description
            input: (todo): write your description
        """
        self._check_device([input, out])
        return self.dispatch(
            [input], [out],
            callback=lambda ws, handle:
                self.feed(ws, handle, starts, sizes),
            no_grad=True,
            check_device=False,
        )


class Cast(function.Function):
    """Cast function."""

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

    def forward(self, input, inplace=False):
        """
        Parameters ---------- input : ndarray.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            inplace: (bool): write your description
        """
        if input.dtype == self.dtype:
            return input
        if inplace:
            return self.dispatch([], [input], no_grad=True)
        return self.dispatch([input], [self.alloc()])


class ChannelAffine(function.Function):
    """ChannelAffine function."""

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

    def forward(self, input, weight, bias=None, out=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            weight: (str): write your description
            bias: (todo): write your description
            out: (array): write your description
        """
        inputs = [input, weight] + ([bias] if bias else [])
        return self.dispatch(inputs, [self.alloc(out)])


class ChannelNormalize(function.Function):
    """ChannelNormalize function."""

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

    def forward(self, input, perm):
        """
        Parse the given input.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            perm: (todo): write your description
        """
        return self.dispatch(
            [input], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, perm),
        )


class ChannelShuffle(function.Function):
    """ChannelShuffle function."""

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
            },
        }

    def forward(self, input, out=None):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input], [self.alloc(out)])


class Concat(function.Function):
    """Concat function."""

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

    def forward(self, seq, out=None):
        """
        Forward sequence.

        Args:
            self: (todo): write your description
            seq: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch(seq, [self.alloc(out)])


class Cumulative(function.Function):
    """Cumulative function."""

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

    def forward(self, input, out=None):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input], [self.alloc(out)])


class Expand(function.Function):
    """Expand function."""

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
        A dict of attributes of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Expand',
            'arguments': {
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'
                    .format(n) for n in range(self.ndim)],
            },
        }

    def feed(self, ws, handle, times):
        """
        Feed data

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            times: (list): write your description
        """
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/dims[{}]'.format(handle, i),
                times[i], 'int64')

    def forward(self, input, dims):
        """
        Perform forward on the layer.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            dims: (todo): write your description
        """
        return self.dispatch(
            [input], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, dims),
        )


class Flatten(function.Function):
    """Flatten function."""

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

    def attributes(self):
        """
        : return : class attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Flatten',
            'arguments': {
                'axis': self.axis,
                'num_axes': self.num_axes,
            }
        }

    def forward(self, input, out=None):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input], [self.alloc(out)])


class IndexSelect(function.Function):
    """IndexSelect function."""

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
            },
        }

    def forward(self, input, index, out=None):
        """
        Parameters ---------- input : array.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            index: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input, index], [self.alloc(out)])


class MaskedAssign(function.Function):
    """MaskedAssign function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(MaskedAssign, self).__init__(key, dev, **kwargs)

    def attributes(self):
        """
        A dict of attributes.

        Args:
            self: (todo): write your description
        """
        return {'op_type': 'MaskedAssign', 'arguments': {}}

    def forward(self, out, mask, input):
        """
        Parameters ---------- out : numpy. ndarray

        Args:
            self: (todo): write your description
            out: (array): write your description
            mask: (todo): write your description
            input: (todo): write your description
        """
        return self.dispatch([input, mask], [self.alloc(out)])


class MaskedSelect(function.Function):
    """MaskedSelect function."""

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

    def forward(self, input, mask, out=None):
        """
        Parameters ---------- inputs.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            mask: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input, mask], [self.alloc(out)])


class Multinomial(function.Function):
    """Multinomial function."""

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
        self.num_samples = kwargs.get('num_samples', 1)

    def attributes(self):
        """
        Returns a dictionary of attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Multinomial',
            'arguments': {
                'epsilon': self.epsilon,
                'normalize': False,
                'num_samples': self.num_samples,
            },
        }

    def forward(self, input, out=None):
        """
        R forward forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input], [self.alloc(out)], no_grad=True)


class NonZero(function.Function):
    """NonZero function."""

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

    def forward(self, input, out=None):
        """
        R forward forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input], [self.alloc(out)], no_grad=True)


class OneHot(function.Function):
    """OneHot function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(OneHot, self).__init__(key, dev, **kwargs)
        self.depth = kwargs.get('depth', 1)

    def attributes(self):
        """
        A dict of the attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'OneHot',
            'arguments': {
                'depth': self.depth,
            },
        }

    def forward(self, input):
        """
        Perform the forward forward.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        return self.dispatch([input], [self.alloc()], no_grad=True)


class Reduce(function.Function):
    """Reduce function."""

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
        Return the attributes for this axis.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Reduce' + self.operation,
            'arguments': {
                'axes': self.axes,
                'keep_dims': self.keep_dims,
            },
        }

    def forward(self, input, out=None):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input], [self.alloc(out)])


class Reshape(function.Function):
    """Reshape function."""

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
        A dict of attributes of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Reshape',
            'arguments': {
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'
                    .format(n) for n in range(self.ndim)],
            },
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
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/dims[{}]'.format(handle, i),
                shape[i], 'int64')

    def forward(self, input, shape, out=None):
        """
        R forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            shape: (int): write your description
            out: (array): write your description
        """
        return self.dispatch(
            [input], [self.alloc(out)],
            callback=lambda ws, handle:
                self.feed(ws, handle, shape),
        )


class Slice(function.Function):
    """Slice function."""

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
            },
        }

    def feed(self, ws, handle, starts, sizes):
        """
        Add the arguments.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            starts: (todo): write your description
            sizes: (int): write your description
        """
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/starts[{}]'.format(handle, i),
                starts[i], 'int64')
            self.feed_arg(
                ws, '{}/sizes[{}]'.format(handle, i),
                sizes[i], 'int64')

    def forward(self, input, starts, sizes):
        """
        Runs the layer.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            starts: (todo): write your description
            sizes: (int): write your description
        """
        return self.dispatch(
            [input], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, starts, sizes)
        )


class Sort(function.Function):
    """Sort function."""

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

    def forward(self, input, outputs=(None, None)):
        """
        Parameters ---------- inputs : list

        Args:
            self: (todo): write your description
            input: (todo): write your description
            outputs: (todo): write your description
        """
        outputs = [self.alloc(outputs[0]), self.alloc(outputs[1])]
        return self.dispatch([input], outputs, no_grad=True)


class Split(function.Function):
    """Split function."""

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

    def attributes(self):
        """
        : return attributes of the axis.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Split',
            'arguments': {
                'axis': self.axis,
                'size_splits': self.size_splits,
            },
        }

    def forward(self, input, chunks):
        """
        Parameters ---------- input.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            chunks: (todo): write your description
        """
        outs = [self.alloc() for _ in range(chunks)]
        return self.dispatch([input], outs)


class Stack(function.Function):
    """Stack function."""

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
        A dictionary of the axis attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Stack',
            'arguments': {
                'axis': self.axis,
            },
        }

    def forward(self, seq, out=None):
        """
        Forward sequence.

        Args:
            self: (todo): write your description
            seq: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch(seq, [self.alloc(out)])


class Squeeze(function.Function):
    """Squeeze function."""

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
        The axes attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Squeeze',
            'arguments': {
                'axes': self.axes,
            },
        }

    def forward(self, input, out=None):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input], [self.alloc(out)])


class Tile(function.Function):
    """Tile function."""

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
        A dict of attributes of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Tile',
            'arguments': {
                'repeats_descs': [
                    '${{HANDLE}}/repeats[{}]'
                    .format(n) for n in range(self.ndim)],
            },
        }

    def feed(self, ws, handle, repeats):
        """
        Feed data.

        Args:
            self: (todo): write your description
            ws: (todo): write your description
            handle: (todo): write your description
            repeats: (int): write your description
        """
        for i in range(self.ndim):
            self.feed_arg(
                ws,
                '{}/repeats[{}]'.format(handle, i),
                repeats[i], 'int64',
            )

    def forward(self, input, times):
        """
        Run a forward forward loop.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            times: (todo): write your description
        """
        return self.dispatch(
            [input], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, times),
        )


class Transpose(function.Function):
    """Transpose function."""

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
        A dict of attributes of attributes

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'Transpose',
            'arguments': {
                'perm_descs': [
                    '${{HANDLE}}/perm[{}]'
                    .format(n) for n in range(self.ndim)],
            },
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

    def forward(self, input, perm):
        """
        Parse the given input.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            perm: (todo): write your description
        """
        return self.dispatch(
            [input], [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, perm),
        )


class TopK(function.Function):
    """TopK function."""

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

    def forward(self, input, outputs=(None, None)):
        """
        Parameters ---------- inputs : list

        Args:
            self: (todo): write your description
            input: (todo): write your description
            outputs: (todo): write your description
        """
        outputs = [self.alloc(outputs[0]), self.alloc(outputs[1])]
        return self.dispatch([input], outputs, no_grad=True)


class Unique(function.Function):
    """Unique function."""

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

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        outputs = [self.alloc() for _ in range(self.num_outputs)]
        return self.dispatch([input], outputs)


class UnSqueeze(function.Function):
    """UnSqueeze function."""

    def __init__(self, key, dev, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            key: (str): write your description
            dev: (todo): write your description
        """
        super(UnSqueeze, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)

    def attributes(self):
        """
        The axes attributes.

        Args:
            self: (todo): write your description
        """
        return {
            'op_type': 'ExpandDims',
            'arguments': {
                'axes': self.axes,
            },
        }

    def forward(self, input, out=None):
        """
        Parse the model.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            out: (array): write your description
        """
        return self.dispatch([input], [self.alloc(out)])


class Where(function.Function):
    """Where function."""

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

    def forward(self, condition, x, y):
        """
        Transforms the given condition.

        Args:
            self: (todo): write your description
            condition: (todo): write your description
            x: (todo): write your description
            y: (todo): write your description
        """
        return self.dispatch([x, y, condition], [self.alloc()])
