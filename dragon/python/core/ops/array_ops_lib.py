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
        super(ArgReduce, self).__init__(key, dev, **kwargs)
        self.op_type = kwargs.get('op_type', 'ArgMax')
        self.axis = kwargs.get('axis', None)
        self.keepdims = kwargs.get('keepdims', True)

    def attributes(self):
        return {
            'op_type': self.op_type,
            'arguments': {
                'axis': self.axis,
                'keepdims': self.keepdims,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class Assign(Operator):
    """Assign operator."""

    def attributes(self):
        return {
            'op_type': 'Assign',
            'arguments': {
                'starts_desc': '${HANDLE}/starts',
                'sizes_desc': '${HANDLE}/sizes',
            },
        }

    def setup(self, ws, handle, starts, sizes):
        self.feed_arg(ws, '%s/starts' % handle, starts, 'int64')
        self.feed_arg(ws, '%s/sizes' % handle, sizes, 'int64')

    def forward(self, inputs, starts, sizes, inplace=False):
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(
            inputs, outputs,
            callback=lambda ws, handle:
                self.setup(ws, handle, starts, sizes),
            no_grad=True,
        )


class Cast(Operator):
    """Cast operator."""

    def __init__(self, key, dev, **kwargs):
        super(Cast, self).__init__(key, dev, **kwargs)
        self.dtype = kwargs.get('dtype', 'float32')

    def attributes(self):
        return {
            'op_type': 'Cast',
            'arguments': {
                'dtype': self.dtype,
            },
        }

    def forward(self, inputs, inplace=False):
        if inputs[0].dtype == self.dtype:
            return inputs[0]
        if inplace:
            return self.dispatch([], [self.alloc(inputs[0])], no_grad=True)
        return self.dispatch(inputs, [self.alloc()])


class ChannelAffine(Operator):
    """ChannelAffine operator."""

    def __init__(self, key, dev, **kwargs):
        super(ChannelAffine, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 1)
        self.num_axes = kwargs.get('num_axes', 1)

    def attributes(self):
        return {
            'op_type': 'ChannelAffine',
            'arguments': {
                'axis': self.axis,
                'num_axes': self.num_axes,
            }
        }

    def forward(self, inputs, inplace=False):
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class ChannelNormalize(Operator):
    """ChannelNormalize operator."""

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
                'perm_desc': '${HANDLE}/perm',
            }
        }

    def setup(self, ws, handle, perm):
        self.feed_arg(ws, '%s/perm' % handle, perm, 'int64')

    def forward(self, inputs, perm):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, perm),
        )


class ChannelShuffle(Operator):
    """ChannelShuffle operator."""

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
    """Concat operator."""

    def __init__(self, key, dev, **kwargs):
        super(Concat, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', 0)

    def attributes(self):
        return {
            'op_type': 'Concat',
            'arguments': {
                'axis': self.axis,
            },
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Cumulative(Operator):
    """Cumulative operator."""

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
    """Expand operator."""

    def __init__(self, key, dev, **kwargs):
        super(Expand, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        return {
            'op_type': 'Expand',
            'arguments': {
                'dims_desc': '${HANDLE}/dims',
            },
        }

    def setup(self, ws, handle, dims):
        self.feed_arg(ws, '%s/dims' % handle, dims, 'int64')

    def forward(self, inputs, dims):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, dims),
        )


class ExpandDims(Operator):
    """ExpandDims operator."""

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
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Flatten(Operator):
    """Flatten operator."""

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
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Identity(Operator):
    """Identity operator."""

    def forward(self, inputs, inplace=False):
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class IndexSelect(Operator):
    """IndexSelect operator."""

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


class LinSpace(Operator):
    """LinSpace operator."""

    def __init__(self, key, dev, **kwargs):
        super(LinSpace, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)
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

    def forward(self, shape, starts, stops, trainable=False):
        out = self.dispatch(
            [], [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, shape, starts, stops),
            no_grad=True,
        )
        out._requires_grad = trainable
        return out


class MaskedAssign(Operator):
    """MaskedAssign operator."""

    def forward(self, inputs, inplace=False):
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs, no_grad=True)


class MaskedSelect(Operator):
    """MaskedSelect operator."""

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Moments(Operator):
    """Moments operator."""

    def __init__(self, key, dev, **kwargs):
        super(Moments, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)
        self.keepdims = kwargs.get('keepdims', True)

    def attributes(self):
        return {
            'op_type': 'Moments',
            'arguments': {
                'axes': self.axes,
                'keepdims': self.keepdims,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc(), self.alloc()])


class Multinomial(Operator):
    """Multinomial operator."""

    def __init__(self, key, dev, **kwargs):
        super(Multinomial, self).__init__(key, dev, **kwargs)
        self.epsilon = kwargs.get('epsilon', 0.)
        self.normalize = kwargs.get('normalize', False)
        self.num_samples = kwargs.get('num_samples', 1)

    def attributes(self):
        return {
            'op_type': 'Multinomial',
            'arguments': {
                'epsilon': self.epsilon,
                'normalize': self.normalize,
                'num_samples': self.num_samples,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class NonZero(Operator):
    """NonZero operator."""

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class OneHot(Operator):
    """OneHot operator."""

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
    """Pad operator."""

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
                'pads_desc': '${HANDLE}/pads',
            }
        }

    def setup(self, ws, handle, pads):
        self.feed_arg(ws, '%s/pads' % handle, pads, 'int64')

    def forward(self, inputs, pads):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, pads),
        )


class Permutation(Operator):
    """Permutation operator."""

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
        self.feed_arg(ws, '%s/limit' % handle, limit, 'int64')

    def forward(self, limit, trainable=False):
        out = self.dispatch(
            [], [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, limit),
            no_grad=True,
        )
        out._requires_grad = trainable
        return out


class Range(Operator):
    """Range operator."""

    def __init__(self, key, dev, **kwargs):
        super(Range, self).__init__(key, dev, **kwargs)
        self.num_args = kwargs.get('num_args', 3)
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

    def forward(self, slice_args, trainable=False):
        out = self.dispatch(
            [], [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, slice_args),
            no_grad=True,
        )
        out._requires_grad = trainable
        return out


class Reduce(Operator):
    """Reduce operator."""

    def __init__(self, key, dev, **kwargs):
        super(Reduce, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)
        self.keepdims = kwargs.get('keepdims', True)
        self.operation = kwargs.get('operation', 'Sum')

    def attributes(self):
        return {
            'op_type': 'Reduce' + self.operation,
            'arguments': {
                'axes': self.axes,
                'keepdims': self.keepdims,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])


class Repeat(Operator):
    """Repeat operator."""

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
    """Reshape operator."""

    def attributes(self):
        return {
            'op_type': 'Reshape',
            'arguments': {
                'dims_desc': '${HANDLE}/dims',
            }
        }

    def setup(self, ws, handle, shape):
        self.feed_arg(ws, '%s/dims' % handle, shape, 'int64')

    def forward(self, inputs, shape, inplace=False):
        return self.dispatch(
            inputs, [self.alloc(inputs[0]) if inplace else self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, shape),
        )


class Slice(Operator):
    """Slice operator."""

    def attributes(self):
        return {
            'op_type': 'Slice',
            'arguments': {
                'starts_desc': '${HANDLE}/starts',
                'sizes_desc': '${HANDLE}/sizes',
            }
        }

    def setup(self, ws, handle, starts, sizes):
        self.feed_arg(ws, '%s/starts' % handle, starts, 'int64')
        self.feed_arg(ws, '%s/sizes' % handle, sizes, 'int64')

    def forward(self, inputs, starts, sizes):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, starts, sizes),
        )


class Shape(Operator):
    """Shape operator."""

    def __init__(self, key, dev, **kwargs):
        super(Shape, self).__init__(key, dev, **kwargs)
        self._device = device_spec.DeviceSpec()

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()], no_grad=True)


class Sort(Operator):
    """Sort operator."""

    def __init__(self, key, dev, **kwargs):
        super(Sort, self).__init__(key, dev, **kwargs)
        self.axis = kwargs.get('axis', -1)
        self.descending = kwargs.get('descending', False)

    def attributes(self):
        return {
            'op_type': 'Sort',
            'arguments': {
                'axis': self.axis,
                'descending': self.descending,
            }
        }

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc(), self.alloc()], no_grad=True)


class Split(Operator):
    """Split operator."""

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
    """Squeeze operator."""

    def __init__(self, key, dev, **kwargs):
        super(Squeeze, self).__init__(key, dev, **kwargs)
        self.axes = kwargs.get('axes', None)

    def attributes(self):
        return {
            'op_type': 'Squeeze',
            'arguments': {'axes': self.axes},
        }

    def forward(self, inputs, inplace=False):
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs)


class Stack(Operator):
    """Stack Operator."""

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
    """Tile operator."""

    def attributes(self):
        return {
            'op_type': 'Tile',
            'arguments': {
                'repeats_desc': '${HANDLE}/repeats',
            }
        }

    def setup(self, ws, handle, repeats):
        self.feed_arg(ws, '%s/repeats' % handle, repeats, 'int64')

    def forward(self, inputs, repeats):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, repeats),
        )


class Transpose(Operator):
    """Transpose operator."""

    def __init__(self, key, dev, **kwargs):
        super(Transpose, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        return {
            'op_type': 'Transpose',
            'arguments': {
                'perm_desc': '${HANDLE}/perm'
                if self.ndim > 0 else None,
            }
        }

    def setup(self, ws, handle, perm):
        if perm is not None:
            self.feed_arg(ws, '%s/perm' % handle, perm, 'int64')

    def forward(self, inputs, perm):
        return self.dispatch(
            inputs, [self.alloc()],
            callback=lambda ws, handle:
                self.setup(ws, handle, perm) if perm else None,
        )


class TopK(Operator):
    """TopK operator."""

    def __init__(self, key, dev, **kwargs):
        super(TopK, self).__init__(key, dev, **kwargs)
        self.k = kwargs.get('k', 1)
        self.axis = kwargs.get('axis', -1)
        self.largest = kwargs.get('largest', True)
        self.sorted = kwargs.get('sorted', True)

    def attributes(self):
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
        return self.dispatch(
            inputs, [self.alloc(), self.alloc()], no_grad=True)


class Unique(Operator):
    """Unique operator."""

    def __init__(self, key, dev, **kwargs):
        super(Unique, self).__init__(key, dev, **kwargs)
        self.return_inverse = kwargs.get('return_inverse', False)
        self.return_counts = kwargs.get('return_counts', False)
        self.num_outputs = 1 + self.return_inverse + self.return_counts

    def attributes(self):
        return {
            'op_type': 'Unique',
            'arguments': {
                'return_inverse': self.return_inverse,
                'return_counts': self.return_counts,
            }
        }

    def forward(self, inputs):
        outputs = [self.alloc() for _ in range(self.num_outputs)]
        return self.dispatch(inputs, outputs)


class Where(Operator):
    """Where operator."""

    def forward(self, inputs):
        return self.dispatch(inputs, [self.alloc()])
