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

"""Define the describing spec for symbolic operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from dragon.core.util import math_util
from dragon.core.util import registry

_GLOBAL_REGISTERED_SPECS = registry.Registry('OpSpec')
get = _GLOBAL_REGISTERED_SPECS.try_get
register = _GLOBAL_REGISTERED_SPECS.register


@register('Accuracy')
def accuracy_spec(args, inputs, outputs):
    _ = locals()
    outputs[0].dtype, outputs[0].shape = 'float32', []
    return outputs


@register('Arange')
def arange_spec(args, inputs, outputs):
    _ = locals()
    outputs[0].dtype = args['dtype']
    slice_args = args['slice']
    if len(slice_args) == 2:
        start, (stop, step) = 0, slice_args
    else:
        start, stop, step = slice_args
    try:
        outputs[0].shape = [int(round((stop - start) / step))]
    except TypeError:
        pass
    return outputs


@register('ArgReduce')
def arg_reduce_spec(args, inputs, outputs):
    outputs[0].dtype = 'int64'
    axis, top_k = args['axis'], args['top_k']
    if args['keep_dims']:
        if axis is None:
            outputs[0].shape = [1]
        else:
            try:
                outputs[0].shape = inputs[0].shape[:]
                if axis < len(outputs[0].shape):
                    outputs[0].shape[axis] = top_k
            except TypeError:
                pass
    else:
        if axis is None:
            if top_k > 1:
                outputs[0].shape = [top_k]
            else:
                outputs[0].shape = []
        else:
            try:
                outputs[0].shape = inputs[0].shape[:]
                if axis < len(outputs[0].shape):
                    if top_k > 1:
                        outputs[0].shape[axis] = top_k
                    else:
                        del outputs[0].shape[axis]
            except TypeError:
                pass
    return outputs


def binary_shape_spec(inputs, outputs):
    if inputs[0].shape is None or inputs[1].shape is None:
        return outputs
    a_shape, b_shape = inputs[0].shape, inputs[1].shape
    y_shape = [None] * max(len(a_shape), len(b_shape))
    i, j, k = len(a_shape) - 1, len(b_shape) - 1, len(y_shape) - 1
    while i >= 0 and j >= 0:
        a_dim, b_dim = a_shape[i], b_shape[j]
        if a_dim is None or b_dim is None:
            y_shape[k] = None
        else:
            y_shape[k] = max(a_dim, b_dim)
        i -= 1
        j -= 1
        k -= 1
    while i >= 0:
        y_shape[k] = a_shape[i]
        i -= 1
        k -= 1
    while j >= 0:
        y_shape[k] = b_shape[j]
        j -= 1
        k -= 1
    outputs[0].shape = y_shape
    return outputs


@register([
    'Add',
    'Div',
    'Maximum',
    'Minimum',
    'Mul',
    'Pow',
    'Sub',
    'Where',
])
def binary_math_spec(args, inputs, outputs):
    _ = locals()
    outputs = binary_shape_spec(inputs, outputs)
    outputs[0].dtype = inputs[0].dtype
    if inputs[0].dtype is None:
        outputs[0].dtype = inputs[1].dtype
    return outputs


@register([
    'Equal',
    'Greater',
    'GreaterEqual',
    'Less',
    'LessEqual',
    'NotEqual',
])
def binary_compare_spec(args, inputs, outputs):
    _ = locals()
    outputs = binary_shape_spec(inputs, outputs)
    outputs[0].dtype = 'bool'
    return outputs


@register('Cast')
def cast_spec(args, inputs, outputs):
    outputs[0].dtype = args['dtype']
    try:
        outputs[0].shape = inputs[0].shape[:]
    except (TypeError, IndexError):
        pass
    return outputs


@register('Concat')
def concat_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    axis = args['axis']
    out_shape = None
    for input in inputs:
        if out_shape is None and input.shape is not None:
            out_shape = input.shape[:]
    try:
        for i in range(len(out_shape)):
            for input in inputs:
                try:
                    if input.shape[i] is not None:
                        out_shape[i] = input.shape[i]
                except (TypeError, IndexError):
                    pass
    except TypeError:
        pass
    try:
        concat_dim = 0
        for input in inputs:
            concat_dim += input.shape[axis]
    except (TypeError, IndexError):
        concat_dim = None
    try:
        out_shape[axis] = concat_dim
        outputs[0].shape = out_shape
    except (TypeError, IndexError):
        outputs[0].shape = None
    return outputs


@register(['Conv2d', 'DepthwiseConv2d'])
def conv_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    out_shape = None
    try:
        out_shape = inputs[0].shape[:]
        channel_axis = 1 if args['data_format'] == 'NCHW' else -1
        spatial_axis = 2 if args['data_format'] == 'NCHW' else 1
        if 'out_channels' in args:
            out_shape[channel_axis] = args['out_channels']
        else:
            out_shape[channel_axis] = inputs[1].shape[0]
        for i in range(len(out_shape) - 2):
            input_size = out_shape[i + spatial_axis]
            k = args['kernel_shape'][i]
            s = args['strides'][i]
            pl, pr = args['pads'][i], args['pads'][i + 2]
            dk, dp = (k - 1) + 1, pl + pr
            if 'SAME' not in args['padding']:
                out_shape[i + spatial_axis] = \
                    int(float(input_size + dp - dk) / s) + 1
            else:
                out_shape[i + spatial_axis] = \
                    int(float(input_size + s - 1) / s)
    except (TypeError, IndexError):
        pass
    outputs[0].shape = out_shape
    return outputs


@register('ConvTranspose2d')
def conv_transpose_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    out_shape = None
    try:
        out_shape = inputs[0].shape[:]
        channel_axis = 1 if args['data_format'] == 'NCHW' else -1
        spatial_axis = 2 if args['data_format'] == 'NCHW' else 1
        if 'out_channels' in args:
            out_shape[channel_axis] = args['out_channels']
        else:
            out_shape[channel_axis] = inputs[1].shape[1]
        for i in range(len(out_shape) - 2):
            k = args['kernel_shape'][i]
            s = args['strides'][i]
            d = args['dilations'][i]
            pl, pr = args['pads'][i], args['pads'][i + 2]
            dk, dp = d * (k - 1) + 1, pl + pr
            input_size = out_shape[i + spatial_axis]
            if 'SAME' not in args['padding']:
                out_shape[i + spatial_axis] = s * \
                    (input_size - 1) + dk - dp
            else:
                out_shape[i + spatial_axis] = None
                if args['output_padding'] is not None:
                    out_shape[i + spatial_axis] = \
                        s * (input_size - 1) + dk + \
                        args['output_padding'][i]
                elif args['output_shape'] is not None:
                    out_shape[i + spatial_axis] = args['output_shape'][i]
    except (TypeError, IndexError):
        pass
    outputs[0].shape = out_shape
    return outputs


@register('DepthToSpace')
def depth_to_space_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    try:
        bs = args['block_size']
        out_shape = inputs[0].shape[:]
        num_axes = len(out_shape) - 2
        if len(out_shape) < 3:
            return outputs
        if args['data_format'] == 'NCHW':
            if out_shape[1] is not None:
                out_shape[1] //= (bs ** num_axes)
            for i in range(2, len(out_shape)):
                if out_shape[i] is not None:
                    out_shape[i] *= bs
        elif args['data_format'] == 'NHWC':
            if out_shape[-1] is not None:
                out_shape[-1] //= (bs ** num_axes)
            for i in range(1, len(out_shape) - 1):
                if out_shape[i] is not None:
                    out_shape[i] *= bs
        outputs[0].shape = out_shape
    except TypeError:
        pass
    return outputs


@register('Dot')
def dot_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    try:
        a_shape, b_shape = inputs[0].shape[:], inputs[1].shape[:]
        if len(a_shape) == 1 and len(b_shape) == 1:
            outputs[0].shape = []
        elif len(a_shape) == 2 and len(b_shape) == 2:
            outputs[0].shape = [a_shape[0], b_shape[1]]
        elif len(a_shape) == 0 and len(b_shape) == 0:
            outputs[0].shape = []
        elif len(a_shape) >= 2 and len(b_shape) == 1:
            outputs[0].shape = a_shape[:-1]
    except TypeError:
        pass
    return outputs


@register([
    'CTCLoss',
    'L1Loss',
    'L2Loss',
    'SigmoidCrossEntropy',
    'SigmoidFocalLoss',
    'SmoothL1Loss',
])
def eltwise_loss_spec(args, inputs, outputs):
    outputs[0].dtype, outputs[0].shape = 'float32', []
    if args['reduction'].upper() == 'NONE':
        try:
            outputs[0].shape = inputs[0].shape[:]
        except TypeError:
            outputs[0].shape = None
    return outputs


@register('Expand')
def expand_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    shape, out_shape = args['dims'], None
    if shape is None:
        return outputs
    try:
        in_shape, out_shape = inputs[0].shape[:], shape[:]
        if len(shape) < len(in_shape):
            num_keep = len(in_shape) - len(shape)
            out_shape = in_shape[:num_keep] + out_shape
        elif len(shape) > len(in_shape):
            num_expand = len(shape) - len(in_shape)
            in_shape = [1] * num_expand + in_shape
        for i, dim in enumerate(out_shape):
            if dim is not None and dim < 0:
                out_shape[i] = in_shape[i]
        outputs[0].shape = out_shape
    except TypeError:
        pass
    return outputs


@register('ExpandDims')
def expand_dims_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    axes = [] if args['axes'] is None else args['axes']
    try:
        out_shape = inputs[0].shape[:] + [0] * len(axes)
        out_rank = len(out_shape)
        for axis in axes:
            while axis < 0:
                axis += out_rank
            if axis < out_rank:
                out_shape[axis] = -1
        j = 0
        for i in range(out_rank):
            if out_shape[i] is not None and \
                    out_shape[i] < 0:
                out_shape[i] = 1
            else:
                if j >= len(inputs[0].shape):
                    break
                out_shape[i] = inputs[0].shape[j]
                j += 1
        outputs[0].shape = list(filter(lambda x: x != 0, out_shape))
    except TypeError:
        pass
    return outputs


@register([
    'Eye',
    'Fill',
    'GlorotNormal',
    'GlorotUniform',
    'RandomNormal',
    'RandomUniform',
    'TruncatedNormal',
])
def fill_spec(args, inputs, outputs):
    _ = locals()
    outputs[0].dtype = args['dtype']
    try:
        if 'dims' in args:
            outputs[0].shape = args['dims'][:]
        else:
            outputs[0].shape = inputs[0].shape[:]
    except (TypeError, KeyError, IndexError):
        pass
    return outputs


@register('Flatten')
def flatten_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    keep_axes = args['keep_axes']
    axis, num_axes = args['axis'], args['num_axes']
    if keep_axes is not None:
        out_shape = [None] * keep_axes
    else:
        out_shape = None
    try:
        in_shape = inputs[0].shape[:]
        if keep_axes is not None:
            if len(in_shape) <= keep_axes:
                out_shape[:len(in_shape)] = in_shape
            else:
                for i in range(keep_axes - 1):
                    out_shape[i] = in_shape[i]
                try:
                    out_shape[keep_axes - 1] = \
                        math_util.prod(in_shape[keep_axes - 1:])
                except (TypeError, IndexError):
                    out_shape[keep_axes - 1] = None
        else:
            if num_axes == -1:
                num_axes = len(in_shape) - axis
            num_axes = max(num_axes, 1)
            try:
                num_flatten = math_util.prod(in_shape[axis:axis + num_axes])
            except TypeError:
                num_flatten = None
            out_shape = \
                in_shape[: axis] + \
                [num_flatten] + \
                in_shape[axis + num_axes:]
    except (TypeError, IndexError):
        pass
    outputs[0].shape = out_shape
    return outputs


@register('FullyConnected')
def fully_connected_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    axis, out_channels = args['axis'], args.get('out_channels', None)
    while axis < 0:
        try:
            axis += len(inputs[0].shape)
        except TypeError:
            return outputs
    outputs[0].shape = [None] * (axis + 1)
    if out_channels is None:
        try:
            if args['transW']:
                out_channels = inputs[1].shape[0]
            else:
                out_channels = inputs[1].shape[1]
        except (TypeError, IndexError):
            out_channels = None
    outputs[0].shape[axis] = out_channels
    try:
        outputs[0].shape[:axis] = inputs[0].shape[:axis]
    except TypeError:
        pass
    return outputs


@register('ChannelNormalize')
def channel_normalize_spec(args, inputs, outputs):
    outputs[0].dtype = args['dtype']
    perm = args['perm']
    if 'perm_desc' in args or 'perm_descs' in args:
        return outputs
    try:
        if perm is None:
            perm = list(range((len(inputs[0].shape))))
        outputs[0].shape = inputs[0].shape[:]
        for i, axis in enumerate(perm):
            outputs[0].shape[i] = inputs[0].shape[axis]
    except (TypeError, IndexError):
        outputs[0].shape = None
    return outputs


@register('IndexSelect')
def index_select_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    axis = args['axis']
    num_axes = args['num_axes']
    try:
        try:
            index_shape = inputs[1].shape[:]
        except TypeError:
            index_shape = [None]
        while axis < 0:
            axis += len(inputs[0].shape)
        out_shape = \
            inputs[0].shape[:axis] + \
            index_shape[:] + \
            inputs[0].shape[axis + num_axes:]
    except TypeError:
        out_shape = None
    outputs[0].shape = out_shape
    return outputs


@register(['IsInf', 'IsNaN'])
def is_spec(args, inputs, outputs):
    _ = locals()
    outputs[0].dtype = 'bool'
    try:
        outputs[0].shape = inputs[0].shape[:]
    except TypeError:
        pass
    return outputs


@register('MaskedSelect')
def masked_select_spec(args, inputs, outputs):
    _ = locals()
    outputs[0].dtype = inputs[0].dtype
    outputs[0].shape = [None]
    return outputs


@register('MatMul')
def matmul_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    ta, tb = args['transA'], args['transB']
    out_shape = None
    try:
        b_shape = inputs[1].shape[:]
        a_shape = out_shape = inputs[0].shape[:]
        out_shape[-2] = a_shape[-1] if ta else a_shape[-2]
        out_shape[-1] = b_shape[-2] if tb else b_shape[-1]
    except TypeError:
        pass
    outputs[0].shape = out_shape
    return outputs


@register('Moments')
def moments_spec(args, inputs, outputs):
    outputs[0].dtype = outputs[1].dtype = \
        inputs[0].dtype if inputs[0].dtype == 'float64' else 'float32'
    axes, keep_dims = args['axes'], args['keep_dims']
    try:
        out_shape = inputs[0].shape[:]
        for axis in axes:
            if axis < len(out_shape):
                out_shape[axis] = 1
        if not keep_dims:
            squeezed_shape = []
            for d in out_shape:
                if d != 1:
                    squeezed_shape.append(d)
            out_shape = squeezed_shape
    except TypeError:
        out_shape = None
    outputs[0].shape = outputs[1].shape = out_shape if axes else []
    return outputs


@register('Multinomial')
def multinomial_spec(args, inputs, outputs):
    outputs[0].dtype = 'int64'
    try:
        outputs[0].shape = inputs[0].shape[:]
        outputs[0].shape[-1] = args['num_samples']
    except TypeError:
        pass
    return outputs


@register('NonZero')
def non_zero_spec(args, inputs, outputs):
    _ = locals()
    outputs[0].dtype = 'int64'
    try:
        outputs[0].shape = [None, len(inputs[0].shape)]
    except TypeError:
        pass
    return outputs


@register('OneHot')
def one_hot_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    try:
        outputs[0].shape = inputs[0].shape[:]
        outputs[0].shape.append(args['depth'])
    except TypeError:
        pass
    return outputs


@register('Pad')
def pad_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    pads, num_dims = args['pads'], len(args['pads']) // 2
    out_shape = None
    try:
        out_shape = inputs[0].shape[:]
        for i in range(num_dims):
            if i < len(out_shape):
                try:
                    out_shape[i] += (pads[i] + pads[i + num_dims])
                except TypeError:
                    out_shape[i] = None
    except TypeError:
        pass
    outputs[0].shape = out_shape
    return outputs


@register('Pool2d')
def pool_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    out_shape = None
    try:
        out_shape = inputs[0].shape[:]
        spatial_axis = 2 if args['data_format'] == 'NCHW' else 1
        for i in range(len(out_shape) - 2):
            k = args['kernel_shape'][i]
            s = args['strides'][i]
            pl, pr = args['pads'][i], args['pads'][i + 2]
            if not args['global_pooling']:
                floor_or_ceil = math.ceil if args['ceil_mode'] else math.floor
                if 'SAME' not in args['padding']:
                    in_size = out_shape[i + spatial_axis] + pl + pr
                    out_size = int(floor_or_ceil(float(in_size - k) / s) + 1)
                    out_shape[i + spatial_axis] = out_size
                else:
                    in_size = out_shape[i + spatial_axis]
                    out_size = int(floor_or_ceil(float(in_size) / s))
                    out_shape[i + spatial_axis] = out_size
            else:
                out_shape[i + spatial_axis] = 1
    except (TypeError, IndexError):
        pass
    outputs[0].shape = out_shape
    return outputs


@register(['PythonPlugin', 'PythonPluginInfer'])
def python_spec(args, inputs, outputs):
    _ = locals()
    return outputs


@register([
    'ReduceMax',
    'ReduceMean',
    'ReduceMin',
    'ReduceSum',
])
def reduce_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    axes, keep_dims = args['axes'], args['keep_dims']
    if axes is None:
        output_shape = []
    else:
        try:
            output_shape = inputs[0].shape[:]
            for axis in axes:
                if axis < len(output_shape):
                    output_shape[axis] = 1
            if not keep_dims:
                squeezed_shape = []
                for d in output_shape:
                    if d != 1:
                        squeezed_shape.append(d)
                output_shape = squeezed_shape
        except TypeError:
            output_shape = None
    outputs[0].shape = output_shape
    return outputs


@register('Repeat')
def repeat_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    axis, repeats = args['axis'], args['repeats']
    if axis is None:
        try:
            num_elements = math_util.prod(inputs[0].shape[:])
            outputs[0].shape = [num_elements * repeats]
        except TypeError:
            outputs[0].shape = [None]
    else:
        try:
            outputs[0].shape = inputs[0].shape[:]
        except TypeError:
            return outputs
        if axis < len(outputs[0].shape):
            try:
                outputs[0].shape[axis] *= repeats
            except TypeError:
                outputs[0].shape[axis] = None
    return outputs


@register('Reshape')
def reshape_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    shape, out_shape = args['dims'], None
    try:
        out_shape = []
        n_elements, n_elements_known = None, None
        try:
            for i, s in enumerate(shape):
                if s == -1:
                    out_shape.append(1)
                elif s == 0:
                    out_shape.append(inputs[0].shape[i])
                else:
                    out_shape.append(s)
        except TypeError:
            out_shape = None
        try:
            n_elements = math_util.prod(inputs[0].shape)
            n_elements_known = math_util.prod(out_shape)
        except TypeError:
            pass
        for i, s in enumerate(shape):
            if s == -1:
                try:
                    out_shape[i] = n_elements // n_elements_known
                except TypeError:
                    out_shape[i] = None
    except TypeError:
        pass
    outputs[0].shape = out_shape
    return outputs


@register('Resize')
def resize_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    if 'sizes_desc' in args or \
       'sizes_descs' in args or \
       'scales_desc' in args or \
       'scales_descs' in args:
        return outputs
    try:
        out_dims = inputs[0].shape[:]
        num_axes = len(out_dims) - 2
        axis = len(out_dims) - 2 if args['data_format'] == 'NCHW' else 1
        try:
            for i in range(num_axes):
                j = axis + i
                if args['sizes'] is not None:
                    if len(args['sizes']) == 1:
                        out_dims[j] = args['sizes'][0]
                    elif len(args['sizes']) == num_axes:
                        out_dims[j] = args['sizes'][i]
                    else:
                        out_dims[j] = args['sizes'][j]
                elif args['scales'] is not None:
                    if len(args['scales']) == 1:
                        out_dims[j] = int(out_dims[j] * args['scales'][0])
                    elif len(args['scales']) == num_axes:
                        out_dims[j] = int(out_dims[j] * args['scales'][i])
                    else:
                        out_dims[j] = int(out_dims[j] * args['sizes'][j])
        except IndexError:
            return outputs
        outputs[0].shape = out_dims
    except TypeError:
        pass
    return outputs


@register(['RoiPool', 'RoiAlign'])
def roi_pool_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    pool_h, pool_w = args['pooled_h'], args['pooled_w']
    out_shape = None
    try:
        out_shape = inputs[0].shape[:]
        out_shape[2:4] = pool_h, pool_w
        try:
            out_shape[0] = inputs[1].shape[0]
        except (TypeError, IndexError):
            out_shape[0] = None
    except TypeError:
        pass
    outputs[0].shape = out_shape
    return outputs


@register('Shape')
def shape_spec(args, inputs, outputs):
    _ = locals()
    outputs[0].dtype = 'int64'
    try:
        outputs[0].shape = [len(inputs[0].shape)]
    except TypeError:
        pass
    return outputs


@register('Slice')
def slice_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    if 'starts_desc' in args or \
       'starts_descs' in args or \
       'sizes_desc' in args or \
       'sizes_descs' in args:
        return outputs
    starts, sizes = args['starts'], args['sizes']
    try:
        in_shape = inputs[0].shape[:]
        ndim = len(in_shape)
        if len(starts) < ndim:
            starts.extend([0] * (ndim - len(starts)))
        out_shape = []
        for i in range(ndim):
            end = in_shape[i]
            size = sizes[i] if i < len(sizes) else -1
            if size > 0:
                out_shape.append(size)
            elif size < 0:
                out_shape.append(None if end is None else end - starts[i])
        outputs[0].shape = out_shape
    except TypeError:
        outputs[0].shape = None
    return outputs


@register([
    'NLLLoss',
    'SoftmaxCrossEntropy',
    'SparseSoftmaxCrossEntropy',
])
def softmax_loss_spec(args, inputs, outputs):
    outputs[0].dtype = 'float32'
    axis, reduction = args['axis'], args['reduction']
    if reduction != 'NONE':
        outputs[0].shape = []
    else:
        try:
            outputs[0].shape = inputs[0].shape[:]
            outputs[0].shape.pop(axis)
        except (TypeError, IndexError):
            pass
    return outputs


@register('SpaceToDepth')
def space_to_depth_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    try:
        bs = args['block_size']
        out_shape = inputs[0].shape[:]
        num_axes = len(out_shape) - 2
        if len(out_shape) < 3:
            return outputs
        if args['data_format'] == 'NCHW':
            if out_shape[1] is not None:
                out_shape[1] *= (bs ** num_axes)
            for i in range(2, len(out_shape)):
                if out_shape[i] is not None:
                    out_shape[i] //= bs
        elif args['data_format'] == 'NHWC':
            if out_shape[-1] is not None:
                out_shape[-1] *= (bs ** num_axes)
            for i in range(1, len(out_shape) - 1):
                if out_shape[i] is not None:
                    out_shape[i] //= bs
        outputs[0].shape = out_shape
    except TypeError:
        pass
    return outputs


@register('Split')
def split_spec(args, inputs, outputs):
    num_outputs = len(outputs)
    for i in range(num_outputs):
        outputs[i].dtype = inputs[0].dtype
    axis = args['axis']
    size_splits = args['size_splits']
    slice_points = args['slice_points']
    if slice_points is not None and len(slice_points) == 0:
        slice_points = None
    slice_offset = 0
    for i in range(len(outputs)):
        try:
            if axis >= len(inputs[0].shape[:]):
                return outputs
            outputs[i].shape = inputs[0].shape[:]
        except TypeError:
            return outputs
        if size_splits is not None:
            try:
                outputs[i].shape[axis] = size_splits[i]
            except IndexError:
                outputs[i].shape[axis] = None
        elif slice_points is not None:
            try:
                if i < len(outputs) - 1:
                    slice_dim = slice_points[i] - slice_offset
                    slice_offset += slice_dim
                else:
                    slice_dim = inputs[0].shape[axis] - slice_offset
            except (TypeError, IndexError):
                slice_dim = None
            outputs[i].shape[axis] = slice_dim
        else:
            try:
                outputs[i].shape[axis] //= num_outputs
            except TypeError:
                outputs[i].shape[axis] = None
    return outputs


@register('Squeeze')
def squeeze_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    axes = [] if args['axes'] is None else args['axes']
    try:
        out_shape = []
        for i, dim in enumerate(inputs[0].shape[:]):
            removed = False
            if dim == 1:
                removed = len(axes) == 0
                for axis in axes:
                    while axis < 0:
                        axis += len(inputs[0].shape)
                    if i == axis:
                        removed = True
            if not removed:
                out_shape.append(dim)
        outputs[0].shape = out_shape
    except TypeError:
        pass
    return outputs


@register('Stack')
def stack_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    axis = args['axis']
    out_shape = None
    for input in inputs:
        if out_shape is None and input.shape is not None:
            out_shape = input.shape[:]
    try:
        for i in range(len(out_shape)):
            for input in inputs:
                try:
                    if input.shape[i] is not None:
                        out_shape[i] = input.shape[i]
                except (TypeError, IndexError):
                    pass
    except TypeError:
        pass
    try:
        while axis < 0:
            axis += (len(out_shape) + 1)
        if axis < 0 or axis >= len(out_shape):
            out_shape.append(len(inputs))
        else:
            out_shape.insert(axis, len(inputs))
        outputs[0].shape = out_shape
    except TypeError:
        outputs[0].shape = None
    return outputs


@register('Tile')
def tile_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    multiples = args['multiples']
    if multiples is not None:
        try:
            out_shape = inputs[0].shape[:]
            for i, multiple in enumerate(multiples):
                if i < len(out_shape):
                    try:
                        out_shape[i] *= multiple
                    except TypeError:
                        out_shape[i] = None
            outputs[0].shape = out_shape
        except TypeError:
            pass
    return outputs


@register('Transpose')
def transpose_spec(args, inputs, outputs):
    outputs[0].dtype = inputs[0].dtype
    perm = args['perm']
    if 'perm_desc' in args or 'perm_descs' in args:
        return outputs
    try:
        if perm is None:
            perm = list(range(((len(inputs[0].shape)) - 1), -1, -1))
        outputs[0].shape = inputs[0].shape[:]
        for i, axis in enumerate(perm):
            outputs[0].shape[i] = inputs[0].shape[axis]
    except (TypeError, IndexError):
        outputs[0].shape = None
    return outputs


@register('Unchanged')
def unchanged_spec(args, inputs, outputs):
    _ = locals()
    outputs[0].dtype = inputs[0].dtype
    try:
        outputs[0].shape = inputs[0].shape[:]
    except TypeError:
        pass
    return outputs
