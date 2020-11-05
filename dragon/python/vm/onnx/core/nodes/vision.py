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

import copy
import numpy

from dragon.vm.onnx.core import exporter
from dragon.vm.onnx.core import helper


@exporter.register([
    'Conv2d',
    'ConvTranspose2d',
    'DepthwiseConv2d',
])
def convolution(op_def, shape_dict, ws):
    """
    Translate a convolution operator.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'ConvTranspose' if 'Transpose' in op_def.type else 'Conv'
    if 'Depthwise' in op_def.type:
        input_shape = shape_dict[op_def.input[0]]
        helper.add_attribute(node, 'group', input_shape[1])
    rank = len(shape_dict[op_def.input[0]]) - 2
    for arg in op_def.arg:
        _assert_data_format(arg)
        if arg.name == 'kernel_shape':
            helper.add_attribute(
                node, 'kernel_shape',
                _normalize_tuple(arg.ints, rank)
            )
        elif arg.name == 'dilations':
            helper.add_attribute(
                node, 'dilations',
                _normalize_tuple(arg.ints, rank)
            )
        elif arg.name == 'strides':
            helper.add_attribute(
                node, 'strides',
                _normalize_tuple(arg.ints, rank)
            )
        elif arg.name == 'pads':
            helper.add_attribute(
                node, 'pads',
                _normalize_pads(arg.ints, rank)
            )
        elif arg.name == 'padding' and arg.s != b'VALID':
            helper.add_attribute(node, 'auto_pad', arg.s)
        elif arg.name == 'group':
            helper.add_attribute(node, 'group', arg.i)
        elif arg.name == 'output_shape':
            helper.add_attribute(node, 'output_shape', arg.ints)
        elif arg.name == 'output_padding':
            helper.add_attribute(node, 'output_padding', arg.ints)
    # Weights and biases
    const_tensors = [helper.from_tensor(e, ws) for e in op_def.input[1:]]
    return node, const_tensors


@exporter.register(['DepthToSpace', 'SpaceToDepth'])
def depth_space_exporter(op_def, shape_dict, ws):
    """
    Return a set of the exporter for the given operation.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    for arg in op_def.arg:
        _assert_data_format(arg)
        if arg.name == 'block_size':
            helper.add_attribute(node, 'blocksize', arg.i)
    return node, const_tensors


@exporter.register('Pool2d')
def pool(op_def, shape_dict, ws):
    """
    Build a pooling operator.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    rank = len(shape_dict[op_def.input[0]]) - 2
    global_pooling, node_copy = 0, copy.deepcopy(node)
    for arg in op_def.arg:
        _assert_data_format(arg)
        if arg.name == 'kernel_shape':
            helper.add_attribute(
                node, 'kernel_shape',
                _normalize_tuple(arg.ints, rank)
            )
        elif arg.name == 'strides':
            helper.add_attribute(
                node, 'strides',
                _normalize_tuple(arg.ints, rank)
            )
        elif arg.name == 'pads':
            helper.add_attribute(
                node, 'pads',
                _normalize_pads(arg.ints, rank)
            )
        elif arg.name == 'padding' and arg.s != b'VALID':
            helper.add_attribute(node, 'auto_pad', arg.s)
        elif arg.name == 'mode':
            if arg.s == b'MAX':
                node.op_type = 'MaxPool'
            elif arg.s == b'AVG':
                node.op_type = 'AveragePool'
        elif arg.name == 'ceil_mode':
            helper.add_attribute(node, 'ceil_mode', arg.i)
        elif arg.name == 'global_pooling':
            global_pooling = arg.i
    if global_pooling > 0:
        # Remove regular pooling attributes.
        node_copy.op_type = 'Global' + node.op_type
        node = node_copy
    return node, const_tensors


@exporter.register('Resize-1')
def resize_v1(op_def, shape_dict, ws):
    """
    Resize v1.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    _ = locals()
    raise RuntimeError('<Upsample> requires opset version >= 7.')


@exporter.register('Resize-7')
def resize_v7(op_def, shape_dict, ws):
    """
    Resize a v7 layer.

    Args:
        op_def: (array): write your description
        shape_dict: (dict): write your description
        ws: (array): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'Upsample'
    input_shape = shape_dict[op_def.input[0]]
    output_shape = shape_dict[op_def.output[0]]
    for arg in op_def.arg:
        if arg.name == 'mode':
            helper.add_attribute(node, 'mode', arg.s.lower())
    helper.add_attribute(
        node, 'scales', [
            float(output_shape[i]) / input_shape[i]
            for i in range(len(input_shape))
        ]
    )
    return node, const_tensors


@exporter.register('Resize-9')
def resize_v9(op_def, shape_dict, ws):
    """
    Resize a v9 operation.

    Args:
        op_def: (array): write your description
        shape_dict: (dict): write your description
        ws: (array): write your description
    """
    node, const_tensors = resize_v10(op_def, shape_dict, ws)
    node.op_type = 'Upsample'
    return node, const_tensors


@exporter.register('Resize-10')
def resize_v10(op_def, shape_dict, ws):
    """
    Resize a v10.

    Args:
        op_def: (array): write your description
        shape_dict: (dict): write your description
        ws: (array): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    input_shape = shape_dict[op_def.input[0]]
    output_shape = shape_dict[op_def.output[0]]
    for arg in op_def.arg:
        if arg.name == 'mode':
            helper.add_attribute(node, 'mode', arg.s.lower())
    scales = helper.from_array(
        numpy.array([
            float(output_shape[i]) / input_shape[i]
            for i in range(len(input_shape))
        ], 'float32'),
        op_def.input[0] + '/resize/scales',
    )
    node.input.extend([scales.name])
    return node, [scales]


@exporter.register('Resize-11')
def resize_v11(op_def, shape_dict, ws):
    """
    Resize a v11 node.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (array): write your description
    """
    node, const_tensors = resize_v10(op_def, shape_dict, ws)
    coord_mode = 'half_pixel'
    for arg in op_def.arg:
        if arg.name == 'mode':
            if arg.s.lower() == b'nearest':
                helper.add_attribute(node, 'nearest_mode', 'floor')
        if arg.name == 'align_corners':
            if arg.i > 0:
                coord_mode = 'align_corners'
    helper.add_attribute(node, 'coordinate_transformation_mode', coord_mode)
    rank = len(shape_dict[op_def.input[0]])
    roi = helper.from_array(
        numpy.array(([0] * rank + [1] * rank), 'float32'),
        op_def.input[0] + '/resize/roi',
    )
    node.input[:] = [node.input[0], roi.name, node.input[1]]
    return node, const_tensors + [roi]


@exporter.register('RoiAlign')
def roi_align(op_def, shape_dict, ws):
    """
    Aligns alignments to alignments.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    # Make a dummy "batch_indices".
    batch_indices = helper.from_array(
        numpy.array([1], 'int64'),
        op_def.input[0] + '/roi_align/batch_indices',
    )
    node.input.extend([batch_indices.name])
    for arg in op_def.arg:
        if arg.name == 'pooled_h':
            helper.add_attribute(node, 'output_height', arg.i)
        elif arg.name == 'pooled_w':
            helper.add_attribute(node, 'output_width', arg.i)
        elif arg.name == 'spatial_scale':
            helper.add_attribute(node, 'spatial_scale', arg.f)
        elif arg.name == 'sampling_ratio':
            helper.add_attribute(node, 'sampling_ratio', arg.i)
    return node, [batch_indices]


@exporter.register('RoiPool')
def roi_pool(op_def, shape_dict, ws):
    """
    Roi pool.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'MaxRoiPool'
    pooled_shape = [None, None]
    for arg in op_def.arg:
        if arg.name == 'pooled_h':
            pooled_shape[0] = arg.i
        elif arg.name == 'pooled_w':
            pooled_shape[1] = arg.i
        elif arg.name == 'spatial_scale':
            helper.add_attribute(node, 'spatial_scale', arg.f)
    helper.add_attribute(node, 'pooled_shape', pooled_shape)
    return node, const_tensors


def _assert_data_format(arg):
    """
    Check if arg is a data format string.

    Args:
        arg: (todo): write your description
    """
    if arg.name == 'data_format':
        if arg.s == 'NHWC':
            raise ValueError('ONNX does not support NHWC format.')


def _normalize_tuple(value, rank):
    """
    Normalize a list.

    Args:
        value: (todo): write your description
        rank: (int): write your description
    """
    if len(value) > rank:
        return [value[i] for i in range(rank)]
    else:
        return [value[i] for i in range(len(value))] + \
               [value[-1] for _ in range(len(value), rank)]


def _normalize_pads(value, rank):
    """
    Normalize a value.

    Args:
        value: (todo): write your description
        rank: (int): write your description
    """
    if len(value) == (rank * 2):
        return value
    return _normalize_tuple(value, rank) * 2
