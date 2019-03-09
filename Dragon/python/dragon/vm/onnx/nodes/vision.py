# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from onnx import numpy_helper
from onnx.helper import make_attribute
from dragon.vm.onnx.nodes.common import CommonONNXExporter


def _assert_data_format(arg):
    if arg.name == 'data_format':
        if arg.s == 'NHWC':
            raise ValueError('ONNX does not support NHWC format.')


def _normalize_tuple(value, rank):
    if len(value) > rank:
        return [value[i] for i in range(rank)]
    else:
        return [value[i] for i in range(len(value))] + \
            [value[-1] for i in range(len(value), rank)]


def _normalize_pads(value, rank):
    if len(value) == (rank * 2): return value
    return _normalize_tuple(value, rank) * 2


def ConvNdONNXExporter(op_def, shape_dict, ws):
    rank = len(shape_dict[op_def.input[0]]) - 2
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    for arg in op_def.arg:
        _assert_data_format(arg)
        if arg.name == 'kernel_shape':
            node_proto.attribute.extend([
                make_attribute('kernel_shape', _normalize_tuple(arg.ints, rank))])
        elif arg.name == 'dilations':
            node_proto.attribute.extend([
                make_attribute('dilations', _normalize_tuple(arg.ints, rank))])
        elif arg.name == 'strides':
            node_proto.attribute.extend([
                make_attribute('strides', _normalize_tuple(arg.ints, rank))])
        elif arg.name == 'pads':
            node_proto.attribute.extend([
                make_attribute('pads', _normalize_pads(arg.ints, rank))])
        elif arg.name == 'padding' and arg.s != b'VALID':
            node_proto.attribute.extend([
                make_attribute('auto_pad', arg.s)])
        elif arg.name == 'group':
            node_proto.attribute.extend([
                make_attribute('group', arg.i)])
        elif arg.name == 'output_shape':
            node_proto.attribute.extend([
                make_attribute('output_shape', arg.ints)])
        elif arg.name == 'output_padding':
            node_proto.attribute.extend([
                make_attribute('output_padding', arg.ints)])

    return node_proto, const_tensors


def PoolNdONNXExporter(op_def, shape_dict, ws):
    rank = len(shape_dict[op_def.input[0]]) - 2
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    for arg in op_def.arg:
        _assert_data_format(arg)
        if arg.name == 'kernel_shape':
            node_proto.attribute.extend([
                make_attribute('kernel_shape', _normalize_tuple(arg.ints, rank))])
        elif arg.name == 'strides':
            node_proto.attribute.extend([
                make_attribute('strides', _normalize_tuple(arg.ints, rank))])
        elif arg.name == 'pads':
            node_proto.attribute.extend([
                make_attribute('pads', _normalize_pads(arg.ints, rank))])
        elif arg.name == 'padding' and arg.s != b'VALID':
            node_proto.attribute.extend([
                make_attribute('auto_pad', arg.s)])
        elif arg.name == 'mode':
            if arg.s == 'MAX': node_proto.op_type = 'MaxPool'
            elif arg.s == 'AVG': node_proto.op_type = 'AveragePool'

    return node_proto, const_tensors


def ResizeNdONNXExporter(op_def, shape_dict, ws):
    input_shape = shape_dict[op_def.input[0]]
    output_shape = shape_dict[op_def.output[0]]

    scales = numpy_helper.from_array(
        np.array([float(output_shape[i]) / input_shape[i]
            for i in range(len(input_shape))], dtype=np.float32),
                name=op_def.input[0] + '/onnx/scales')

    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    node_proto.attribute.extend([
        make_attribute('mode', {
            'NNResize': 'nearest',
            'BilinearResize': 'bilinear',
        }[op_def.type])])
    node_proto.input.extend([scales.name])

    return node_proto, [scales]


def ROIPoolONNXExporter(op_def, shape_dict, ws):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)
    pooled_shape = [None, None]

    for arg in op_def.arg:
        if arg.name == 'pool_h':
            pooled_shape[0] = arg.i
        elif arg.name == 'pool_w':
            pooled_shape[1] = arg.i
        elif arg.name == 'spatial_scale':
            node_proto.attribute.extend([
                make_attribute('spatial_scale', arg.f)])

    node_proto.attribute.extend([
        make_attribute('pooled_shape', pooled_shape)])

    return node_proto, const_tensors


def ROIAlignONNXExporter(op_def, shape_dict, ws):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)
    node_proto.op_type = 'ATen' # Template
    node_proto.attribute.extend([make_attribute('op_type', 'ROIAlign')])

    for arg in op_def.arg:
        if arg.name == 'pool_h':
            node_proto.attribute.extend([
                make_attribute('pool_h', arg.i)])
        elif arg.name == 'pool_w':
            node_proto.attribute.extend([
                make_attribute('pool_w', arg.i)])
        elif arg.name == 'spatial_scale':
            node_proto.attribute.extend([
                make_attribute('spatial_scale', arg.f)])
        elif arg.name == 'sampling_ratio':
            node_proto.attribute.extend([
                make_attribute('sampling_ratio', arg.i)])

    return node_proto, const_tensors