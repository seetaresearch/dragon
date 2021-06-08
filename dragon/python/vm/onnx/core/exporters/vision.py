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

from dragon.vm.onnx.core import helper
from dragon.vm.onnx.core.exporters import utils as export_util


@export_util.register([
    'Conv',
    'ConvTranspose',
    'DepthwiseConv',
])
def conv_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    node.op_type = 'ConvTranspose' if 'Transpose' in op_def.type else 'Conv'
    if 'Depthwise' in op_def.type:
        input_shape = context.blob_shapes[op_def.input[0]]
        helper.add_attribute(node, 'group', input_shape[1])
    rank = len(context.blob_shapes[op_def.input[0]]) - 2
    for arg in op_def.arg:
        _assert_data_format(arg)
        if arg.name == 'kernel_shape':
            helper.add_attribute(
                node, 'kernel_shape',
                _normalize_tuple(arg.ints, rank))
        elif arg.name == 'dilations':
            helper.add_attribute(
                node, 'dilations',
                _normalize_tuple(arg.ints, rank))
        elif arg.name == 'strides':
            helper.add_attribute(
                node, 'strides',
                _normalize_tuple(arg.ints, rank))
        elif arg.name == 'pads':
            helper.add_attribute(
                node, 'pads',
                _normalize_pads(arg.ints, rank))
        elif arg.name == 'padding' and arg.s != b'VALID':
            helper.add_attribute(node, 'auto_pad', arg.s)
        elif arg.name == 'group':
            helper.add_attribute(node, 'group', arg.i)
        elif arg.name == 'output_shape':
            helper.add_attribute(node, 'output_shape', arg.ints)
        elif arg.name == 'output_padding':
            helper.add_attribute(node, 'output_padding', arg.ints)
    return node, const_tensors


@export_util.register(['DepthToSpace', 'SpaceToDepth'])
def depth_space_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    for arg in op_def.arg:
        _assert_data_format(arg)
        if arg.name == 'block_size':
            helper.add_attribute(node, 'blocksize', arg.i)
        elif arg.name == 'mode':
            if node.op_type != 'SpaceToDepth':
                helper.add_attribute(node, 'mode', arg.s)
    return node, const_tensors


@export_util.register('Pool')
def pool(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    rank = len(context.blob_shapes[op_def.input[0]]) - 2
    global_pool, node_copy = 0, copy.deepcopy(node)
    for arg in op_def.arg:
        _assert_data_format(arg)
        if arg.name == 'kernel_shape':
            helper.add_attribute(
                node, 'kernel_shape',
                _normalize_tuple(arg.ints, rank))
        elif arg.name == 'strides':
            helper.add_attribute(
                node, 'strides',
                _normalize_tuple(arg.ints, rank))
        elif arg.name == 'pads':
            helper.add_attribute(
                node, 'pads',
                _normalize_pads(arg.ints, rank))
        elif arg.name == 'padding' and arg.s != b'VALID':
            helper.add_attribute(node, 'auto_pad', arg.s)
        elif arg.name == 'mode':
            if arg.s == b'MAX':
                node.op_type = 'MaxPool'
            elif arg.s == b'AVG':
                node.op_type = 'AveragePool'
        elif arg.name == 'ceil_mode':
            helper.add_attribute(node, 'ceil_mode', arg.i)
        elif arg.name == 'global_pool':
            global_pool = arg.i
    if global_pool > 0:
        # Remove regular pooling attributes.
        node_copy.op_type = 'Global' + node.op_type
        node = node_copy
    return node, const_tensors


@export_util.register('Resize-1')
def resize_v1(op_def, context):
    raise RuntimeError('<Upsample> requires opset version >= 7.')


@export_util.register('Resize-7')
def resize_v7(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    node.op_type = 'Upsample'
    input_shape = context.blob_shapes[op_def.input[0]]
    output_shape = context.blob_shapes[op_def.output[0]]
    for arg in op_def.arg:
        if arg.name == 'mode':
            helper.add_attribute(node, 'mode', arg.s.lower())
    helper.add_attribute(
        node, 'scales', [float(output_shape[i]) / input_shape[i]
                         for i in range(len(input_shape))])
    return node, const_tensors


@export_util.register('Resize-9')
def resize_v9(op_def, context):
    node, const_tensors = resize_v10(**locals())
    node.op_type = 'Upsample'
    return node, const_tensors


@export_util.register('Resize-10')
def resize_v10(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    input_shape = context.blob_shapes[op_def.input[0]]
    output_shape = context.blob_shapes[op_def.output[0]]
    for arg in op_def.arg:
        if arg.name == 'mode':
            helper.add_attribute(node, 'mode', arg.s.lower())
    scales = helper.from_array(
        numpy.array([float(output_shape[i]) / input_shape[i]
                     for i in range(len(input_shape))], 'float32'),
        context.unique_name(op_def.input[0] + '/resize/scales'),
    )
    node.input.extend([scales.name])
    return node, [scales]


@export_util.register('Resize-11')
def resize_v11(op_def, context):
    node, const_tensors = resize_v10(**locals())
    coord_mode = 'half_pixel'
    for arg in op_def.arg:
        if arg.name == 'mode':
            if arg.s.lower() == b'nearest':
                helper.add_attribute(node, 'nearest_mode', 'floor')
        if arg.name == 'align_corners':
            if arg.i > 0:
                coord_mode = 'align_corners'
    helper.add_attribute(node, 'coordinate_transformation_mode', coord_mode)
    rank = len(context.blob_shapes[op_def.input[0]])
    roi = helper.from_array(
        numpy.array(([0] * rank + [1] * rank), 'float32'),
        context.unique_name(op_def.input[0] + '/resize/roi'),
    )
    node.input[:] = [node.input[0], roi.name, node.input[1]]
    return node, const_tensors + [roi]


@export_util.register('RoiAlign')
def roi_align(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    # Make a dummy "batch_indices".
    batch_indices = helper.from_array(
        numpy.zeros((context.blob_shapes[node.input[1]][0],), 'int64'),
        context.unique_name(op_def.input[0] + '/roi_align/batch_indices'),
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


@export_util.register('RoiPool')
def roi_pool(op_def, context):
    node, const_tensors = export_util.translate(**locals())
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
    if arg.name == 'data_format':
        if arg.s == 'NHWC':
            raise ValueError('ONNX does not support NHWC format.')


def _normalize_tuple(value, rank):
    if len(value) > rank:
        return [value[i] for i in range(rank)]
    else:
        return [value[i] for i in range(len(value))] + \
               [value[-1] for _ in range(len(value), rank)]


def _normalize_pads(value, rank):
    if len(value) == (rank * 2):
        return value
    return _normalize_tuple(value, rank) * 2
