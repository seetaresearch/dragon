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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
try:
    from onnx import TensorProto
except ImportError:
    from dragon.core.util import deprecation
    TensorProto = deprecation.not_installed('ONNX')

from dragon.vm.onnx.core import exporter
from dragon.vm.onnx.core import helper


@exporter.register(['ArgMax', 'ArgMin'])
def arg_reduce_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    # ONNX requires indices only, remove the values.
    indices = node.output[0]
    node.ClearField('output')
    node.output.extend([indices])
    for arg in op_def.arg:
        if arg.name == 'axis':
            helper.add_attribute(node, 'axis', arg.i)
        elif arg.name == 'keep_dims':
            helper.add_attribute(node, 'keepdims', arg.i)
    return node, None


@exporter.register('Cast')
def cast_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'Cast'
    if len(node.input) == 0:
        raise ValueError('ONNX does not support in-place cast.')
    for arg in op_def.arg:
        if arg.name == 'dtype':
            helper.add_attribute(node, 'dtype', helper.tensor_type(arg.s))
    return node, const_tensors


@exporter.register('ChannelNormalize')
def channel_normalize_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'ATen'  # Currently not supported in ai.onnx
    helper.add_attribute(node, 'op_type', 'ChannelNormalize')
    for arg in op_def.arg:
        if arg.name == 'mean':
            helper.add_attribute(node, 'mean', arg.floats)
        elif arg.name == 'std':
            helper.add_attribute(node, 'std', arg.floats)
        elif arg.name == 'axis':
            helper.add_attribute(node, 'axis', arg.i)
        elif arg.name == 'dtype':
            helper.add_attribute(node, 'dtype', arg.s)
        elif arg.name == 'perm':
            helper.add_attribute(node, 'perm', arg.ints)
        elif arg.name == 'perm_desc':
            values = helper.fetch_argument(op_def, arg, ws)
            helper.add_attribute(node, 'perm', values)
        elif arg.name == 'perm_descs':
            if len(arg.strings) > 0:
                values = helper.fetch_arguments(op_def, arg, ws)
                helper.add_attribute(node, 'perm', values)
    return node, const_tensors


@exporter.register('Concat')
def concat_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    for arg in op_def.arg:
        if arg.name == 'axis':
            helper.add_attribute(node, 'axis', arg.i)
    return node, const_tensors


@exporter.register('CumSum')
def cumulative_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    axis = 0
    for arg in op_def.arg:
        if arg.name == 'axis':
            axis = arg.i
        elif arg.name == 'exclusive':
            helper.add_attribute(node, 'exclusive', arg.i)
        elif arg.name == 'reverse':
            helper.add_attribute(node, 'reverse', arg.i)
    axis = helper.from_array(
        numpy.array(axis, 'int64'),
        op_def.input[0] + '/cumulative/axis',
    )
    node.input.extend([axis.name])
    return node, [axis]


@exporter.register('Expand')
def expand_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    shape = list(shape_dict[op_def.output[0]])
    shape = helper.from_array(
        numpy.array(shape, 'int64'),
        op_def.input[0] + '/expand/shape',
    )
    node.input.extend([shape.name])
    return node, [shape]


@exporter.register('ExpandDims')
def expand_dims_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'Unsqueeze'
    axes = None
    for arg in op_def.arg:
        if arg.name == 'axes':
            axes = arg.ints
    if axes is not None:
        helper.add_attribute(node, 'axes', axes)
    return node, const_tensors


@exporter.register('Eye')
def eye_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    if len(op_def.input) > 0:
        node.op_type += 'Like'
    else:
        output_shape = list(shape_dict[op_def.output[0]])
        helper.add_attribute(node, 'shape', output_shape)
    for arg in op_def.arg:
        if arg.name == 'k':
            helper.add_attribute(node, 'k', arg.i)
        elif arg.name == 'dtype':
            helper.add_attribute(node, 'dtype', helper.tensor_type(arg.s))
    return node, const_tensors


@exporter.register('Flatten')
def flatten_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    for arg in op_def.arg:
        if arg.name == 'axis':
            helper.add_attribute(node, 'axis', arg.i)
        elif arg.name == 'num_axes':
            if arg.i != -1:
                raise ValueError(
                    'Excepted <num_axes> is -1, '
                    'got {}.'.format(arg.i)
                )
        elif arg.name == 'keep_axes':
            raise ValueError('<keep_axes> should not be set.')
    return node, None


@exporter.register('IndexSelect')
def index_select_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'Gather'
    for arg in op_def.arg:
        if arg.name == 'axis':
            helper.add_attribute(node, 'axis', arg.i)
        elif arg.name == 'num_axes':
            if arg.i > 1:
                raise ValueError('Reshape to avoid selecting multiple axes.')
    return node, const_tensors


@exporter.register('Multinomial')
def multinomial_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    helper.add_attribute(node, 'dtype', helper.tensor_type('int64'))
    for arg in op_def.arg:
        if arg.name == 'num_samples':
            helper.add_attribute(node, 'sample_size', arg.i)
    return node, const_tensors


@exporter.register('OneHot')
def one_hot_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    helper.add_attribute(node, 'axis', -1)
    depth, on_value, off_value = 1, 1, 0
    dtype = ws.FetchTensor(node.output[0]).dtype
    for arg in op_def.arg:
        if arg.name == 'depth':
            depth = arg.i
        elif arg.name == 'on_value':
            on_value = arg.i
        elif arg.name == 'off_value':
            off_value = arg.i
    depth = helper.from_array(
        numpy.array(depth, 'int64'),
        op_def.input[0] + '/one_hot/depth',
    )
    values = helper.from_array(
        numpy.array([off_value, on_value], dtype),
        op_def.input[0] + '/one_hot/values',
    )
    const_tensors = [depth, values]
    node.input.extend([depth.name, values.name])
    return node, const_tensors


def pad_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    pads, value = [], 0
    for arg in op_def.arg:
        if arg.name == 'pads':
            pads = [int(e) for e in arg.ints]
        elif arg.name == 'pads_desc':
            pads = helper.fetch_argument(op_def, arg, ws)
        elif arg.name == 'pads_descs':
            pads = helper.fetch_arguments(op_def, arg, ws)
        elif arg.name == 'mode':
            helper.add_attribute(node, 'mode', arg.s.lower())
        elif arg.name == 'value':
            value = arg.f
    return node, pads, value


@exporter.register('Pad-1')
def pad_exporter_v1(op_def, shape_dict, ws):
    node, pads, value = pad_exporter(**locals())
    helper.add_attribute(node, 'paddings', pads)
    helper.add_attribute(node, 'value', value)
    return node, []


@exporter.register('Pad-2')
def pad_exporter_v2(op_def, shape_dict, ws):
    node, pads, value = pad_exporter(**locals())
    helper.add_attribute(node, 'pads', pads)
    helper.add_attribute(node, 'value', value)
    return node, []


@exporter.register('Pad-11')
def pad_exporter_v11(op_def, shape_dict, ws):
    node, pads, value = pad_exporter(**locals())
    pads = helper.from_array(
        numpy.array(pads, 'int64'),
        op_def.input[0] + '/pad/pads',
    )
    value = helper.from_array(
        numpy.array(value, 'float64'),
        op_def.input[0] + '/pad/value',
    )
    node.input.extend([pads.name, value.name])
    return node, [pads, value]


@exporter.register('RandomNormal')
def random_normal_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    if len(op_def.input) > 0:
        node.op_type += 'Like'
    else:
        output_shape = list(shape_dict[op_def.output[0]])
        helper.add_attribute(node, 'shape', output_shape)
    for arg in op_def.arg:
        if arg.name == 'mean':
            helper.add_attribute(node, 'mean', arg.f)
        elif arg.name == 'std':
            helper.add_attribute(node, 'scale', arg.f)
        elif arg.name == 'dtype':
            helper.add_attribute(node, 'dtype', helper.tensor_type(arg.s))
    return node, const_tensors


@exporter.register('RandomUniform')
def random_uniform_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    if len(op_def.input) > 0:
        node.op_type += 'Like'
    else:
        output_shape = list(shape_dict[op_def.output[0]])
        helper.add_attribute(node, 'shape', output_shape)
    for arg in op_def.arg:
        if arg.name == 'low':
            helper.add_attribute(node, 'low', arg.f)
        elif arg.name == 'high':
            helper.add_attribute(node, 'high', arg.f)
        elif arg.name == 'dtype':
            helper.add_attribute(node, 'dtype', helper.tensor_type(arg.s))
    return node, const_tensors


@exporter.register(['ReduceMax', 'ReduceMean', 'ReduceMin', 'ReduceSum'])
def reduce_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    axes = list(range(len(shape_dict[op_def.input[0]])))
    for arg in op_def.arg:
        if arg.name == 'axes':
            axes = arg.ints
        elif arg.name == 'keep_dims':
            helper.add_attribute(node, 'keepdims', arg.i)
    helper.add_attribute(node, 'axes', axes)
    return node, const_tensors


@exporter.register('Reshape')
def reshape_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    shape = dims = list(shape_dict[op_def.output[0]])
    for arg in op_def.arg:
        if arg.name == 'dims':
            dims = [int(e) for e in arg.ints]
        elif arg.name == 'dims_desc':
            dims = helper.fetch_argument(op_def, arg, ws)
        elif arg.name == 'dims_descs':
            dims = helper.fetch_arguments(op_def, arg, ws)
    for axis, dim in enumerate(dims):
        shape[axis] = dim if dim <= 0 else shape[axis]
    shape = helper.from_array(
        numpy.array(shape, 'int64'),
        op_def.input[0] + '/reshape/shape',
    )
    node.input.extend([shape.name])
    return node, [shape]


def slice_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    in_shape = shape_dict[op_def.input[0]]
    starts, sizes, ends = [], [], []
    for arg in op_def.arg:
        if arg.name == 'starts':
            starts = [int(e) for e in arg.ints]
        elif arg.name == 'starts_desc':
            starts = helper.fetch_argument(op_def, arg, ws)
        elif arg.name == 'starts_descs':
            starts = helper.fetch_arguments(op_def, arg, ws)
        elif arg.name == 'sizes':
            sizes = [int(e) for e in arg.ints]
        elif arg.name == 'sizes_desc':
            sizes = helper.fetch_argument(op_def, arg, ws)
        elif arg.name == 'sizes_descs':
            sizes = helper.fetch_arguments(op_def, arg, ws)
    for i, size in enumerate(sizes):
        if size == -1:
            ends.append(in_shape[i])
        elif size == 0:
            ends.append(starts[i] + 1)
        else:
            ends.append(starts[i] + size)
    return node, starts, ends


@exporter.register('Slice-1')
def slice_exporter_v1(op_def, shape_dict, ws):
    node, starts, ends = slice_exporter(**locals())
    helper.add_attribute(node, 'axes', numpy.arange(len(starts)))
    helper.add_attribute(node, 'ends', ends)
    helper.add_attribute(node, 'starts', starts)
    return node, []


@exporter.register('Slice-10')
def slice_exporter_v10(op_def, shape_dict, ws):
    node, starts, ends = slice_exporter(**locals())
    axes = helper.from_array(
        numpy.arange(len(starts), dtype='int64'),
        op_def.input[0] + '/slice/axes',
    )
    starts = helper.from_array(
        numpy.array(starts, 'int64'),
        op_def.input[0] + '/slice/starts',
    )
    ends = helper.from_array(
        numpy.array(ends, 'int64'),
        op_def.input[0] + '/slice/ends',
    )
    node.input.extend([starts.name, ends.name, axes.name])
    return node, [starts, ends, axes]


@exporter.register('Split')
def split_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    axis = 0
    for arg in op_def.arg:
        if arg.name == 'axis':
            axis = arg.i
    size_splits = [shape_dict[e][axis] for e in op_def.output]
    helper.add_attribute(node, 'split', size_splits)
    return node, const_tensors


@exporter.register('Squeeze')
def squeeze_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    axes = None
    for arg in op_def.arg:
        if arg.name == 'axes':
            axes = arg.ints
    if axes is not None:
        helper.add_attribute(node, 'axes', axes)
    return node, const_tensors


@exporter.register('Tile')
def tile_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    repeats = []
    for arg in op_def.arg:
        if arg.name == 'repeats':
            repeats = [e for e in arg.ints]
        elif arg.name == 'repeats_desc':
            repeats = helper.fetch_argument(op_def, arg, ws)
        elif arg.name == 'repeats_descs':
            repeats = helper.fetch_arguments(op_def, arg, ws)
    repeats = helper.from_array(
        numpy.array(repeats, 'int64'),
        op_def.input[0] + '/tile/repeats',
    )
    node.input.extend([repeats.name])
    return node, [repeats]


@exporter.register('Transpose')
def transpose_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    for arg in op_def.arg:
        if arg.name == 'perm':
            helper.add_attribute(node, 'perm', arg.ints)
        elif arg.name == 'perm_desc':
            values = helper.fetch_argument(op_def, arg, ws)
            helper.add_attribute(node, 'perm', values)
        elif arg.name == 'perm_descs':
            if len(arg.strings) > 0:
                values = helper.fetch_arguments(op_def, arg, ws)
                helper.add_attribute(node, 'perm', values)
    return node, None


def top_k_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    k, axis, largest, sorted = 1, -1, True, True
    for arg in op_def.arg:
        if arg.name == 'k':
            k = arg.i
        if arg.name == 'axis':
            axis = arg.i
        elif arg.name == 'largest':
            largest = arg.i
        elif arg.name == 'sorted':
            sorted = arg.i
    return node, (k, axis, largest, sorted)


@exporter.register('TopK-1')
def top_k_exporter_v1(op_def, shape_dict, ws):
    node, (k, axis, largest, sorted) = top_k_exporter(op_def, shape_dict, ws)
    if largest == 0:
        raise ValueError('TopK-1 does not support smallest mode.')
    helper.add_attribute(node, 'axis', axis)
    helper.add_attribute(node, 'k', k)
    return node, None


@exporter.register('TopK-10')
def top_k_exporter_v10(op_def, shape_dict, ws):
    node, (k, axis, largest, sorted) = top_k_exporter(op_def, shape_dict, ws)
    if largest == 0:
        raise ValueError('TopK-10 does not support smallest mode.')
    helper.add_attribute(node, 'axis', axis)
    k = helper.from_array(
        numpy.array([k], 'int64'),
        op_def.input[0] + '/top_k/k',
    )
    node.input.extend([k.name])
    return node, [k]


@exporter.register('TopK-11')
def top_k_exporter_v11(op_def, shape_dict, ws):
    node, (k, axis, largest, sorted) = top_k_exporter(op_def, shape_dict, ws)
    helper.add_attribute(node, 'axis', axis)
    helper.add_attribute(node, 'largest', largest)
    helper.add_attribute(node, 'sorted', sorted)
    k = helper.from_array(
        numpy.array([k], 'int64'),
        op_def.input[0] + '/top_k/k',
    )
    node.input.extend([k.name])
    return node, [k]
