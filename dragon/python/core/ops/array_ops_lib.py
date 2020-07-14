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
"""The array ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Arange(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Arange, self).__init__(key, dev, **kwargs)
        self.num_args = kwargs.get('num_args', 3)
        self.dtype = kwargs.get('dtype', 'int64')

    def attributes(self):
        return {
            'op_type': 'Arange',
            'arguments': {
                'dtype': self.dtype,
                'slice_descs': [
                    '${{HANDLE}}/slice[{}]'
                    .format(n) for n in range(self.num_args)],
            }
        }

    def feed(self, ws, handle, slice_args):
        for i in range(len(slice_args)):
            self.feed_arg(
                ws, '{}/slice[{}]'.format(handle, i),
                slice_args[i], 'float32')

    def forward(self, slice_args, trainable=False):
        output = self.dispatch(
            [], [self.alloc()],
            callback=lambda ws, handle:
            self.feed(ws, handle, slice_args)
        )
        output._requires_grad = trainable
        return output


class ArgReduce(Operator):
    def __init__(self, key, dev, **kwargs):
        super(ArgReduce, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', None)
        self.top_k = kwargs.get('top_k', 1)
        self.keep_dims = kwargs.get('keep_dims', True)
        self.operation = kwargs.get('operation', 'MAX')

    def attributes(self):
        return {
            'op_type': 'ArgReduce',
            'arguments': {
                'axis': self.axis,
                'top_k': self.top_k,
                'keep_dims': self.keep_dims,
                'operation': self.operation,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class Cast(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Cast, self).__init__(key, dev, **kwargs)
        self.dtype = kwargs.get('dtype', 'float32')

    def attributes(self):
        return {
            'op_type': 'Cast',
            'arguments': {'dtype': self.dtype},
        }

    def forward(self, inputs, inplace=False):
        if inputs[0].dtype == self.dtype:
            return inputs[0]
        if inplace:
            return self.dispatch([], inputs, no_grad=True)
        return self.dispatch(inputs, [self.alloc()])


class ChannelNormalize(Operator):
    def __init__(self, key, dev, **kwargs):
        super(ChannelNormalize, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)
        self.ndim = kwargs.get('ndim', 0)
        self.mean = kwargs.get('mean', None)
        self.std = kwargs.get('std', None)
        self.dtype = kwargs.get('dtype', 'float32')

    def attributes(self):
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
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/perm[{}]'.format(handle, i),
                perm[i], 'int64')

    def forward(self, inputs, perm):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
            self.feed(ws, handle, perm)
        )


class ChannelShuffle(Operator):
    def __init__(self, key, dev, **kwargs):
        super(ChannelShuffle, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.group = kwargs.get('group', 1)

    def attributes(self):
        return {
            'op_type': 'ChannelShuffle',
            'arguments': {
                'axis': self.axis,
                'group': self.group,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Concat(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Concat, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)

    def attributes(self):
        return {
            'op_type': 'Concat',
            'arguments': {'axis': self.axis},
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Cumulative(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Cumulative, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.exclusive = kwargs.get('exclusive', False)
        self.reverse = kwargs.get('reverse', False)
        self.operation = kwargs.get('operation', 'Sum')

    def attributes(self):
        return {
            'op_type': 'Cum' + self.operation,
            'arguments': {
                'axis': self.axis,
                'exclusive': self.exclusive,
                'reverse': self.reverse,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Expand(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Expand, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        return {
            'op_type': 'Expand',
            'arguments': {
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, dims):
        for i, dim in enumerate(dims):
            self.feed_arg(
                ws, '{}/dims[{}]'.format(handle, i),
                dim, 'int64')

    def forward(self, inputs, dims):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, dims),
        )


class ExpandDims(Operator):
    def __init__(self, key, dev, **kwargs):
        super(ExpandDims, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', [0])

    def attributes(self):
        return {
            'op_type': 'ExpandDims',
            'arguments': {
                'axes': self.axes,
            }
        }

    def forward(self, inputs, inplace=False):
        outputs = [inputs[0] if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Flatten(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Flatten, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.num_axes = kwargs.get('num_axes', -1)
        self.keep_axes = kwargs.get('keep_axes', None)

    def attributes(self):
        return {
            'op_type': 'Flatten',
            'arguments': {
                'axis': self.axis,
                'num_axes': self.num_axes,
                'keep_axes': self.keep_axes,
            }
        }

    def forward(self, inputs, inplace=False):
        outputs = [inputs[0] if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class IndexSelect(Operator):
    def __init__(self, key, dev, **kwargs):
        super(IndexSelect, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.num_axes = kwargs.get('num_axes', 1)

    def attributes(self):
        return {
            'op_type': 'IndexSelect',
            'arguments': {
                'axis': self.axis,
                'num_axes': self.num_axes,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class MaskedSelect(Operator):
    def __init__(self, key, dev, **kwargs):
        super(MaskedSelect, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {'op_type': 'MaskedSelect', 'arguments': {}}

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Moments(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Moments, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)
        self.keep_dims = kwargs.get('keep_dims', True)

    def attributes(self):
        return {
            'op_type': 'Moments',
            'arguments': {
                'axes': self.axes,
                'keep_dims': self.keep_dims,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc(), self.alloc()])


class Multinomial(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Multinomial, self).__init__(key, dev, **kwargs)
        self.eps = kwargs.get('eps', 0.)
        self.normalize = kwargs.get('normalize', False)
        self.num_samples = kwargs.get('num_samples', 1)

    def attributes(self):
        return {
            'op_type': 'Multinomial',
            'arguments': {
                'eps': self.eps,
                'normalize': self.normalize,
                'num_samples': self.num_samples,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class NonZero(Operator):
    def __init__(self, key, dev, **kwargs):
        super(NonZero, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {'op_type': 'NonZero', 'arguments': {}}

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class OneHot(Operator):
    def __init__(self, key, dev, **kwargs):
        super(OneHot, self).__init__(key, dev, **kwargs)
        self.depth = kwargs.get('depth', 1)
        self.on_value = kwargs.get('on_value', 1)
        self.off_value = kwargs.get('off_value', 0)

    def attributes(self):
        return {
            'op_type': 'OneHot',
            'arguments': {
                'depth': self.depth,
                'on_value': self.on_value,
                'off_value': self.off_value,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class Pad(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Pad, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)
        self.value = kwargs.get('value', 0.)
        self.mode = kwargs.get('mode', 'CONSTANT')

    def attributes(self):
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
        for i, e in enumerate(pads):
            self.feed_arg(
                ws, '{}/pads[{}]'.format(handle, i),
                e, 'int64')

    def forward(self, inputs, pads):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, pads),
        )


class Reduce(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Reduce, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)
        self.keep_dims = kwargs.get('keep_dims', True)
        self.operation = kwargs.get('operation', 'Sum')

    def attributes(self):
        return {
            'op_type': 'Reduce' + self.operation,
            'arguments': {
                'axes': self.axes,
                'keep_dims': self.keep_dims,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Repeat(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Repeat, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 2147483647)
        self.repeats = kwargs.get('repeats', 1)

    def attributes(self):
        return {
            'op_type': 'Repeat',
            'arguments': {
                'axis': self.axis,
                'repeats': self.repeats,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Reshape(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Reshape, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        return {
            'op_type': 'Reshape',
            'arguments': {
                'dims_descs': [
                    '${{HANDLE}}/dims[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, shape):
        for i, e in enumerate(shape):
            self.feed_arg(
                ws, '{}/dims[{}]'.format(handle, i),
                e, 'int64')

    def forward(self, inputs, shape, inplace=False):
        outputs = [inputs[0] if inplace else self.alloc()]
        return self.dispatch(
            inputs, outputs,
            callback=lambda ws, handle:
            self.feed(ws, handle, shape)
        )


class Slice(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Slice, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
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
        for i in range(len(starts)):
            self.feed_arg(
                ws, '{}/starts[{}]'.format(handle, i),
                starts[i], 'int64')
            self.feed_arg(
                ws, '{}/sizes[{}]'.format(handle, i),
                sizes[i], 'int64')

    def forward(self, inputs, starts, sizes):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, starts, sizes),
        )


class Shape(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Shape, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {'op_type': 'Shape', 'arguments': {}}

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Split(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Split, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)
        self.size_splits = kwargs.get('size_splits', None)
        self.slice_points = kwargs.get('slice_points', None)

    def attributes(self):
        return {
            'op_type': 'Split',
            'arguments': {
                'axis': self.axis,
                'size_splits': self.size_splits,
                'slice_points': self.slice_points,
            }
        }

    def forward(self, inputs, chunks):
        outputs = [self.alloc() for _ in range(chunks)]
        return self.dispatch(inputs, outputs)


class Squeeze(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Squeeze, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)

    def attributes(self):
        return {
            'op_type': 'Squeeze',
            'arguments': {'axes': self.axes},
        }

    def forward(self, inputs, inplace=False):
        outputs = [inputs[0] if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Stack(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Stack, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)

    def attributes(self):
        return {
            'op_type': 'Stack',
            'arguments': {'axis': self.axis},
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Tile(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Tile, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        return {
            'op_type': 'Tile',
            'arguments': {
                'repeats_descs': [
                    '${{HANDLE}}/repeats[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, repeats):
        for i, size in enumerate(repeats):
            self.feed_arg(
                ws, '{}/repeats[{}]'.format(handle, i),
                size, 'int64')

    def forward(self, inputs, repeats):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, repeats),
        )


class Transpose(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Transpose, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        return {
            'op_type': 'Transpose',
            'arguments': {
                'perm_descs': [
                    '${{HANDLE}}/perm[{}]'
                    .format(n) for n in range(self.ndim)],
            }
        }

    def feed(self, ws, handle, perm):
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/perm[{}]'.format(handle, i),
                perm[i], 'int64')

    def forward(self, inputs, perm):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.feed(ws, handle, perm),
        )


class Where(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Where, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {'op_type': 'Where', 'arguments': {}}

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])
