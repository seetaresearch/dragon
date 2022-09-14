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
"""Shape and data type inference for symbols."""

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
    outputs[0]._dtype, outputs[0]._shape = 'float32', ()
    return outputs


@register(['ArgMax', 'ArgMin'])
def arg_reduce_spec(args, inputs, outputs):
    outputs[0]._dtype = 'int64'
    axis = args['axis']
    if args['keepdims']:
        try:
            out_shape = list(inputs[0].shape[:])
            out_shape[axis] = 1
            outputs[0]._shape = tuple(out_shape)
        except (TypeError, IndexError):
            pass
    else:
        try:
            out_shape = list(inputs[0].shape[:])
            if axis < len(out_shape):
                del out_shape[axis]
            outputs[0]._shape = tuple(out_shape)
        except (TypeError, IndexError):
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
    outputs[0]._shape = tuple(y_shape)
    return outputs


@register(['Add',
           'Atan2',
           'BitwiseAnd',
           'BitwiseOr',
           'BitwiseXor',
           'Div',
           'Maximum',
           'Minimum',
           'Mul',
           'Pow',
           'Sub'])
def binary_math_spec(args, inputs, outputs):
    outputs = binary_shape_spec(inputs, outputs)
    outputs[0]._dtype = inputs[0].dtype
    if inputs[0].dtype is None:
        outputs[0]._dtype = inputs[1].dtype
    return outputs


@register(['And',
           'Or',
           'Xor',
           'Equal',
           'Greater',
           'GreaterEqual',
           'Less',
           'LessEqual',
           'NotEqual'])
def binary_compare_spec(args, inputs, outputs):
    outputs = binary_shape_spec(inputs, outputs)
    outputs[0]._dtype = 'bool'
    return outputs


@register('BooleanMask')
def boolean_mask_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    outputs[0]._shape = (None,)
    return outputs


@register('Cast')
def cast_spec(args, inputs, outputs):
    outputs[0]._dtype = args['dtype']
    try:
        outputs[0]._shape = inputs[0].shape[:]
    except (TypeError, IndexError):
        pass
    return outputs


@register('Concat')
def concat_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    axis = args['axis']
    out_shape = None
    for input in inputs:
        if out_shape is None and input.shape is not None:
            out_shape = list(input.shape[:])
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
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register(['Conv', 'DepthwiseConv'])
def conv_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        out_shape = list(inputs[0].shape[:])
        num_axes = len(out_shape) - 2
        channel_axis = 1 if args['data_format'] == 'NCHW' else -1
        spatial_axis = 2 if args['data_format'] == 'NCHW' else 1
        if 'out_channels' in args:
            out_shape[channel_axis] = args['out_channels']
        else:
            out_shape[channel_axis] = inputs[1].shape[0]
        for i in range(num_axes):
            try:
                k = args['kernel_shape'][i]
                s = args['strides'][i]
                d = args['dilations'][i]
                in_size = out_shape[i + spatial_axis]
                k_size = d * (k - 1) + 1
                if 'SAME' not in args['padding']:
                    pad_size = args['pads'][i] + args['pads'][i + num_axes]
                    out_size = (in_size + pad_size - k_size) // s + 1
                else:
                    out_size = (in_size + s - 1) // s
            except (IndexError, TypeError):
                out_size = None
            out_shape[i + spatial_axis] = out_size
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register('ConvTranspose')
def conv_transpose_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        out_shape = list(inputs[0].shape[:])
        num_axes = len(out_shape) - 2
        channel_axis = 1 if args['data_format'] == 'NCHW' else -1
        spatial_axis = 2 if args['data_format'] == 'NCHW' else 1
        if 'out_channels' in args:
            out_shape[channel_axis] = args['out_channels']
        else:
            out_shape[channel_axis] = inputs[1].shape[1]
        for i in range(num_axes):
            if ('output_padding_desc' in args or
                    'output_shape_desc' in args):
                out_shape[i + spatial_axis] = None
                continue
            try:
                k = args['kernel_shape'][i]
                s = args['strides'][i]
                d = args['dilations'][i]
                in_size = out_shape[i + spatial_axis]
                k_size = d * (k - 1) + 1
                out_size = None
                if 'SAME' not in args['padding']:
                    pad_size = args['pads'][i] + args['pads'][i + num_axes]
                    out_size = s * (in_size - 1) + k_size - pad_size
                    if 'output_padding' in args and args['output_padding']:
                        out_size += args['output_padding'][i]
                else:
                    if 'output_shape' in args and args['output_shape']:
                        out_size = args['output_shape'][i]
                        if 'output_padding' in args and args['output_padding']:
                            out_size += args['output_padding'][i]
            except (IndexError, TypeError):
                out_size = None
            out_shape[i + spatial_axis] = out_size
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register('DepthToSpace')
def depth_to_space_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        bs = args['block_size']
        out_shape = list(inputs[0].shape[:])
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
        outputs[0]._shape = tuple(out_shape)
    except TypeError:
        pass
    return outputs


@register(['CTCLoss',
           'L1Loss',
           'L2Loss',
           'SigmoidCrossEntropyLoss',
           'SigmoidFocalLoss',
           'SmoothL1Loss'])
def eltwise_loss_spec(args, inputs, outputs):
    outputs[0]._dtype, outputs[0]._shape = 'float32', ()
    if args['reduction'].upper() == 'NONE':
        try:
            outputs[0]._shape = inputs[0].shape[:]
        except TypeError:
            outputs[0]._shape = None
    return outputs


@register('Expand')
def expand_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        out_shape = None
        if 'dims_desc' in args:
            out_shape = [None] * len(inputs[0].shape)
        elif 'dims' in args:
            in_shape = list(inputs[0].shape[:])
            dims = args['dims']
            out_shape = list(dims[:])
            if len(dims) < len(in_shape):
                num_keep = len(in_shape) - len(dims)
                out_shape = in_shape[:num_keep] + out_shape
            elif len(dims) > len(in_shape):
                num_expand = len(dims) - len(in_shape)
                in_shape = [1] * num_expand + in_shape
                for i, dim in enumerate(out_shape):
                    if dim is not None and dim < 0:
                        out_shape[i] = in_shape[i]
        outputs[0]._shape = tuple(out_shape)
    except (KeyError, TypeError):
        outputs[0]._shape = None
    return outputs


@register('Im2Col')
def im2col_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        out_shape = list(inputs[0].shape[:])
        num_axes = len(out_shape) - 2
        channel_axis = 1 if args['data_format'] == 'NCHW' else -1
        spatial_axis = 2 if args['data_format'] == 'NCHW' else 1
        for i in range(num_axes):
            try:
                k = args['kernel_shape'][i]
                s = args['strides'][i]
                d = args['dilations'][i]
                in_size = out_shape[i + spatial_axis]
                k_size = d * (k - 1) + 1
                if 'SAME' not in args['padding']:
                    pad_size = args['pads'][i] + args['pads'][i + num_axes]
                    out_size = (in_size + pad_size - k_size) // s + 1
                else:
                    out_size = (in_size + s - 1) // s
                if out_shape[channel_axis] is not None:
                    out_shape[channel_axis] *= k
            except (IndexError, TypeError):
                out_size = None
            out_shape[i + spatial_axis] = out_size
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register(['Eye',
           'Fill',
           'GlorotNormal',
           'GlorotUniform',
           'RandomNormal',
           'RandomUniform',
           'TruncatedNormal'])
def fill_spec(args, inputs, outputs):
    outputs[0]._dtype = args['dtype']
    try:
        if 'dims' in args:
            outputs[0]._shape = tuple(args['dims'][:])
        else:
            outputs[0]._shape = tuple(inputs[0].shape[:])
    except (TypeError, KeyError, IndexError):
        outputs[0]._shape = None
    return outputs


@register('Flatten')
def flatten_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    axis = args['axis']
    end_axis = args.get('end_axis', None)
    end_axis = axis if end_axis is None else end_axis
    try:
        in_shape = list(inputs[0].shape[:])
        out_shape = in_shape[: axis]
        num_axes = len(in_shape[axis:end_axis]) + 1
        try:
            out_shape += [math_util.prod(in_shape[axis:axis + num_axes])]
        except TypeError:
            out_shape += [None]
        out_shape += in_shape[axis + num_axes:]
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register('Gemm')
def gemm_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    out_shape = []
    try:
        if args['transA']:
            out_shape.append(inputs[0].shape[-1])
        else:
            out_shape.extend(inputs[0].shape[:-1])
    except (TypeError, IndexError):
        return outputs
    try:
        if args['transB']:
            out_shape.extend(inputs[1].shape[:-1])
        else:
            out_shape.append(inputs[1].shape[1])
    except (TypeError, IndexError):
        return outputs
    outputs[0]._shape = tuple(out_shape)
    return outputs


@register('ChannelNorm')
def channel_normalize_spec(args, inputs, outputs):
    outputs[0]._dtype = args['dtype']
    try:
        out_shape = list(inputs[0].shape[:])
        if 'perm' in args:
            perm = args['perm']
            if perm is None:
                perm = list(range(len(inputs[0].shape)))
            out_shape = list(inputs[0].shape[:])
            for i, axis in enumerate(perm):
                out_shape[i] = inputs[0].shape[axis]
        else:
            out_shape = [None] * len(out_shape)
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register('Gather')
def gather_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    axis = args['axis']
    end_axis = args.get('end_axis', None)
    end_axis = axis if end_axis is None else end_axis
    try:
        try:
            index_shape = inputs[1].shape[:]
        except TypeError:
            index_shape = [None]
        num_axes = len(inputs[0].shape[axis:end_axis]) + 1
        out_shape = \
            inputs[0].shape[:axis] + \
            index_shape[:] + \
            inputs[0].shape[axis + num_axes:]
    except TypeError:
        out_shape = None
    outputs[0]._shape = out_shape
    return outputs


@register('GatherElements')
def gather_elements_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        outputs[0]._shape = inputs[1]._shape[:]
    except TypeError:
        pass
    return outputs


@register(['IsFinite', 'IsInf', 'IsNaN', 'Not'])
def is_spec(args, inputs, outputs):
    outputs[0]._dtype = 'bool'
    try:
        outputs[0]._shape = inputs[0].shape[:]
    except TypeError:
        pass
    return outputs


@register('LinSpace')
def linspace_spec(args, inputs, outputs):
    outputs[0]._dtype = args['dtype']
    outputs[0]._shape = tuple(args['dims'])
    return outputs


@register('MatMul')
def matmul_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        a_shape = list(inputs[0].shape[:])
        b_shape = list(inputs[1].shape[:])
        if len(a_shape) >= 2 and len(b_shape) >= 2:
            out_shape = [1] * max(len(a_shape), len(b_shape))
            a_shape = [1] * (len(out_shape) - len(a_shape)) + a_shape
            b_shape = [1] * (len(out_shape) - len(b_shape)) + b_shape
            for i in range(len(out_shape)):
                try:
                    out_shape[i] = max(a_shape[i], b_shape[i])
                except TypeError:
                    out_shape[i] = None
            out_shape[-2] = a_shape[-2]
            out_shape[-1] = b_shape[-1]
        elif len(a_shape) == 1 and len(b_shape) == 1:
            out_shape = []
        else:
            out_shape = a_shape if len(b_shape) == 1 else b_shape
            out_shape.pop(-1 if len(b_shape) == 1 else -2)
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register('Moments')
def moments_spec(args, inputs, outputs):
    out_dtype = 'float32'
    if inputs[0].dtype == 'float64':
        out_dtype = 'float64'
    elif inputs[0].dtype == 'int64':
        out_dtype = 'float64'
    outputs[0]._dtype = outputs[1]._dtype = out_dtype
    axes, keepdims = args['axes'], args['keepdims']
    try:
        out_shape = list(inputs[0].shape[:])
        for axis in axes:
            if axis < len(out_shape):
                out_shape[axis] = -1
        if not keepdims:
            squeezed_shape = []
            for d in out_shape:
                if d >= 0:
                    squeezed_shape.append(d)
            out_shape = squeezed_shape
        else:
            out_shape = [1 if d < 0 else d for d in out_shape]
    except TypeError:
        if axes is None:
            out_shape = (1,) if keepdims else ()
        else:
            out_shape = None
    out_shape = tuple(out_shape) if out_shape is not None else None
    outputs[0]._shape = outputs[1]._shape = out_shape
    return outputs


@register('Multinomial')
def multinomial_spec(args, inputs, outputs):
    outputs[0]._dtype = 'int64'
    try:
        out_shape = list(inputs[0].shape[:])
        out_shape[-1] = args['num_samples']
        outputs[0]._shape = tuple(out_shape)
    except TypeError:
        outputs[0]._shape = None
    return outputs


@register('NonZero')
def non_zero_spec(args, inputs, outputs):
    outputs[0]._dtype = 'int64'
    try:
        outputs[0]._shape = (None, len(inputs[0].shape))
    except TypeError:
        pass
    return outputs


@register('OneHot')
def one_hot_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        outputs[0]._shape = inputs[0].shape[:] + (args['depth'],)
    except TypeError:
        pass
    return outputs


@register('Pad')
def pad_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    pads, num_dims = args['pads'], len(args['pads']) // 2
    out_shape = None
    try:
        out_shape = list(inputs[0].shape[:])
        for i in range(num_dims):
            if i < len(out_shape):
                try:
                    out_shape[i] += (pads[i] + pads[i + num_dims])
                except TypeError:
                    out_shape[i] = None
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register('Permutation')
def permutation_spec(args, inputs, outputs):
    outputs[0]._dtype = args['dtype']
    if 'limit_desc' in args:
        outputs[0]._shape = (None,)
    else:
        outputs[0]._shape = (args['limit'],)
    return outputs


@register('Pool')
def pool_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    out_shape = None
    try:
        out_shape = list(inputs[0].shape[:])
        num_axes = len(out_shape) - 2
        spatial_axis = 2 if args['data_format'] == 'NCHW' else 1
        for i in range(num_axes):
            if not args['global_pool']:
                try:
                    k = args['kernel_shape'][i]
                    s = args['strides'][i]
                    in_size = out_shape[i + spatial_axis]
                    if 'SAME' not in args['padding']:
                        floor_or_ceil = math.ceil if args['ceil_mode'] else math.floor
                        pad_size = args['pads'][i] + args['pads'][i + num_axes]
                        out_size = float(in_size + pad_size - k) / float(s) + 1
                        out_size = floor_or_ceil(out_size)
                    else:
                        out_size = math.ceil(float(in_size) / float(s))
                except TypeError:
                    out_size = None
                out_shape[i + spatial_axis] = out_size
            else:
                out_shape[i + spatial_axis] = 1
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register(['PythonPlugin', 'PythonPluginInfer'])
def python_spec(args, inputs, outputs):
    return outputs


@register('Range')
def range_spec(args, inputs, outputs):
    outputs[0]._dtype = args['dtype']
    slice_args = args['slice']
    if len(slice_args) == 2:
        start, (limit, delta) = 0, slice_args
    else:
        start, limit, delta = slice_args
    try:
        outputs[0]._shape = (int(math.ceil((limit - start) / delta)),)
    except (TypeError, ZeroDivisionError):
        pass
    return outputs


@register(['ReduceMax',
           'ReduceMean',
           'ReduceMin',
           'ReduceSum',
           'ReduceVar',
           'ReduceL1',
           'ReduceL2'])
def reduce_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    axes, keepdims = args['axes'], args['keepdims']
    if axes is None:
        outputs[0]._shape = ()
    else:
        try:
            out_shape = list(inputs[0].shape[:])
            for axis in axes:
                out_shape[axis] = -1
            if not keepdims:
                squeezed_shape = []
                for d in out_shape:
                    if d != -1:
                        squeezed_shape.append(d)
                out_shape = squeezed_shape
            else:
                out_shape = [1 if d < 0 else d for d in out_shape]
            outputs[0]._shape = tuple(out_shape)
        except (TypeError, IndexError):
            outputs[0]._shape = None
    return outputs


@register('Repeat')
def repeat_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    if 'repeats_desc' in args:
        return outputs
    axis, repeats = args['axis'], args['repeats']
    if axis is None:
        try:
            num_elements = math_util.prod(inputs[0].shape[:])
            outputs[0]._shape = (num_elements * repeats,)
        except TypeError:
            outputs[0]._shape = (None,)
    else:
        try:
            out_shape = list(inputs[0].shape[:])
        except TypeError:
            return outputs
        if axis < len(out_shape):
            try:
                out_shape[axis] *= repeats
            except TypeError:
                out_shape[axis] = None
        outputs[0]._shape = tuple(out_shape)
    return outputs


@register('Reshape')
def reshape_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        shape = args['dims']
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
        except IndexError:
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
    except (KeyError, TypeError):
        out_shape = None
    outputs[0]._shape = tuple(out_shape) if out_shape is not None else None
    return outputs


@register('Resize')
def resize_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        out_shape = list(inputs[0].shape[:])
        if 'sizes_desc' in args or 'scales_desc' in args:
            outputs[0]._shape = (None,) * len(out_shape)
            return outputs
        num_axes = len(out_shape) - 2
        axis = len(out_shape) - 2 if args['data_format'] == 'NCHW' else 1
        try:
            for i in range(num_axes):
                j = axis + i
                if args['sizes'] is not None:
                    if len(args['sizes']) == 1:
                        out_shape[j] = args['sizes'][0]
                    elif len(args['sizes']) == num_axes:
                        out_shape[j] = args['sizes'][i]
                    else:
                        out_shape[j] = args['sizes'][j]
                elif args['scales'] is not None:
                    try:
                        if len(args['scales']) == 1:
                            out_shape[j] = int(out_shape[j] * args['scales'][0])
                        elif len(args['scales']) == num_axes:
                            out_shape[j] = int(out_shape[j] * args['scales'][i])
                        else:
                            out_shape[j] = int(out_shape[j] * args['scales'][j])
                    except TypeError:
                        out_shape[j] = None
        except IndexError:
            return outputs
        outputs[0]._shape = tuple(out_shape)
    except TypeError:
        outputs[0]._shape = None
    return outputs


@register(['RoiPool', 'RoiAlign'])
def roi_pool_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    pool_h, pool_w = args['pooled_h'], args['pooled_w']
    out_shape = None
    try:
        out_shape = list(inputs[0].shape[:])
        out_shape[2:4] = pool_h, pool_w
        try:
            out_shape[0] = inputs[1].shape[0]
        except (TypeError, IndexError):
            out_shape[0] = None
        outputs[0]._shape = tuple(out_shape)
    except TypeError:
        outputs[0]._shape = None
    return outputs


@register('Shape')
def shape_spec(args, inputs, outputs):
    outputs[0]._dtype = 'int64'
    try:
        outputs[0]._shape = (len(inputs[0].shape),)
    except TypeError:
        outputs[0]._shape = None
    return outputs


@register('Slice')
def slice_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    if 'starts_desc' in args or 'sizes_desc' in args:
        return outputs
    starts, sizes = list(args['starts']), list(args['sizes'])
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
        outputs[0]._shape = tuple(out_shape)
    except TypeError:
        outputs[0]._shape = None
    return outputs


@register(['NLLLoss', 'SoftmaxCrossEntropyLoss'])
def softmax_loss_spec(args, inputs, outputs):
    outputs[0]._dtype = 'float32'
    axis, reduction = args['axis'], args['reduction']
    if reduction.upper() != 'NONE':
        outputs[0]._shape = ()
    else:
        try:
            out_shape = list(inputs[0].shape[:])
            out_shape.pop(axis)
            outputs[0]._shape = tuple(out_shape)
        except (TypeError, IndexError):
            pass
    return outputs


@register('Sort')
def sort_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    outputs[1]._dtype = 'int64'
    try:
        out_shape = list(inputs[0].shape[:])
        outputs[0]._shape = tuple(out_shape[:])
        outputs[1]._shape = tuple(out_shape[:])
    except (TypeError, IndexError):
        outputs[0]._shape = outputs[1]._shape = None
    return outputs


@register('SpaceToDepth')
def space_to_depth_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        bs = args['block_size']
        out_shape = list(inputs[0].shape[:])
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
        outputs[0]._shape = tuple(out_shape)
    except TypeError:
        pass
    return outputs


@register('Split')
def split_spec(args, inputs, outputs):
    num_outputs = len(outputs)
    for i in range(num_outputs):
        outputs[i]._dtype = inputs[0].dtype
    axis = args['axis']
    keepdims = args.get('keepdims', True)
    size_splits = args.get('split', None)
    for i in range(len(outputs)):
        try:
            if axis >= len(inputs[0].shape[:]):
                return outputs
            out_shape = list(inputs[0].shape[:])
        except TypeError:
            return outputs
        if 'split_desc' in args:
            out_shape[axis] = None
        elif size_splits is not None:
            out_shape[axis] = size_splits[i]
        else:
            try:
                slice_dim = (out_shape[axis] + num_outputs - 1) // num_outputs
                if i == num_outputs - 1:
                    slice_dim = out_shape[axis] - slice_dim * (num_outputs - 1)
                out_shape[axis] = slice_dim
            except TypeError:
                out_shape[axis] = None
        if out_shape is not None:
            if not keepdims:
                out_shape.pop(axis)
            outputs[i]._shape = tuple(out_shape)
    return outputs


@register('Squeeze')
def squeeze_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
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
        outputs[0]._shape = tuple(out_shape)
    except TypeError:
        outputs[0]._shape = None
    return outputs


@register('Stack')
def stack_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    axis = args['axis']
    out_shape = None
    for input in inputs:
        if out_shape is None and input.shape is not None:
            out_shape = list(input.shape[:])
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
        outputs[0]._shape = tuple(out_shape)
    except TypeError:
        outputs[0]._shape = None
    return outputs


@register('Tile')
def tile_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        out_shape = list(inputs[0].shape[:])
        if 'repeats' in args:
            repeats = args['repeats']
            num_repeats = len(repeats)
            num_dims = max(num_repeats, len(out_shape))
            start_axis = num_dims - len(out_shape)
            out_shape = [1] * start_axis + out_shape[:]
            for i, size in enumerate(repeats):
                start_axis = i + (num_dims - num_repeats)
                try:
                    out_shape[start_axis] *= size
                except TypeError:
                    out_shape[start_axis] = None
        else:
            out_shape = [None] * len(out_shape)
        outputs[0]._shape = tuple(out_shape)
    except (KeyError, TypeError):
        outputs[0]._shape = None
    return outputs


@register('Transpose')
def transpose_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        out_shape = list(inputs[0].shape[:])
        if 'perm' in args:
            perm = args['perm']
            if perm is None:
                perm = list(range(((len(inputs[0].shape)) - 1), -1, -1))
            out_shape = list(inputs[0].shape[:])
            for i, axis in enumerate(perm):
                out_shape[i] = inputs[0].shape[axis]
        else:
            out_shape = [None] * len(out_shape)
        outputs[0]._shape = tuple(out_shape)
    except (TypeError, IndexError):
        outputs[0]._shape = None
    return outputs


@register('TopK')
def top_k_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    outputs[1]._dtype = 'int64'
    k, axis = args['k'], args['axis']
    axis = -1 if axis is None else axis
    try:
        out_shape = list(inputs[0].shape[:])
        out_shape[axis] = k
        outputs[0]._shape = tuple(out_shape[:])
        outputs[1]._shape = tuple(out_shape[:])
    except (TypeError, IndexError):
        outputs[0]._shape = outputs[1]._shape = None
    return outputs


@register('Unchanged')
def unchanged_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    try:
        outputs[0]._shape = inputs[0].shape[:]
    except TypeError:
        pass
    return outputs


@register('Unique')
def unique_spec(args, inputs, outputs):
    return_inverse = args['return_inverse']
    return_counts = args['return_counts']
    outputs[0]._dtype = inputs[0].dtype
    for i in range(1, len(outputs)):
        outputs[i]._dtype = 'int64'
    outputs[0]._shape = (None,)
    if len(outputs) == 2:
        if return_inverse:
            try:
                outputs[1]._shape = inputs[0].shape[:]
            except TypeError:
                pass
        elif return_counts:
            outputs[1]._shape = (None,)
    elif len(outputs) == 3:
        try:
            outputs[1]._shape = inputs[0].shape[:]
        except TypeError:
            pass
        outputs[2]._shape = (None,)
    return outputs


@register('Unsqueeze')
def unsqueeze_spec(args, inputs, outputs):
    outputs[0]._dtype = inputs[0].dtype
    axes = [] if args['axes'] is None else args['axes']
    try:
        out_shape = list(inputs[0].shape[:]) + [0] * len(axes)
        out_rank = len(out_shape)
        for axis in axes:
            while axis < 0:
                axis += out_rank
            if axis < out_rank:
                out_shape[axis] = -1
        j = 0
        for i in range(out_rank):
            if out_shape[i] is not None and out_shape[i] < 0:
                out_shape[i] = 1
            else:
                if j >= len(inputs[0].shape):
                    break
                out_shape[i] = inputs[0].shape[j]
                j += 1
        outputs[0]._shape = tuple(filter(lambda x: x != 0, out_shape))
    except TypeError:
        pass
    return outputs


@register('Where')
def where_spec(args, inputs, outputs):
    outputs = binary_shape_spec(inputs[1:], outputs)
    outputs[0]._dtype = inputs[1].dtype
    if inputs[1].dtype is None:
        outputs[0]._dtype = inputs[2].dtype
    return outputs
