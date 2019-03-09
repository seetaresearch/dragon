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

from onnx import TensorProto
from onnx.helper import make_attribute

from dragon.vm.onnx.nodes.common import CommonONNXExporter


def ImageDataONNXExporter(op_def, shape_dict, ws):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)
    node_proto.op_type = 'ATen' # Template
    node_proto.attribute.extend([make_attribute('op_type', 'ImageData')])

    for arg in op_def.arg:
        if arg.name == 'mean_values':
            node_proto.attribute.extend([
                make_attribute('mean_values', arg.floats)])
        elif arg.name == 'std_values':
            node_proto.attribute.extend([
                make_attribute('std_values', arg.floats)])
        elif arg.name == 'dtype':
            node_proto.attribute.extend([
                make_attribute('dtype', arg.s)])
        elif arg.name == 'data_format':
            node_proto.attribute.extend([
                make_attribute('data_format', arg.s)])

    return node_proto, const_tensors


def AsTypeONNXExporter(op_def, shape_dict, ws):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)
    node_proto.op_type = 'Cast'

    if len(node_proto.input) == 0:
        raise ValueError('ONNX does not support in-place cast.')

    for arg in op_def.arg:
        if arg.name == 'dtype':
            if arg.s.upper() == b'BOOL':
                node_proto.attribute.extend([
                    make_attribute('to', TensorProto.BOOL)])
            elif arg.s.upper() == b'INT8':
                node_proto.attribute.extend([
                    make_attribute('to', TensorProto.INT8)])
            elif arg.s.upper() == b'UINT8':
                node_proto.attribute.extend([
                    make_attribute('to', TensorProto.UINT8)])
            elif arg.s.upper() == b'INT32':
                node_proto.attribute.extend([
                    make_attribute('to', TensorProto.INT32)])
            elif arg.s.upper() == b'INT64':
                node_proto.attribute.extend([
                    make_attribute('to', TensorProto.INT64)])
            elif arg.s.upper() == b'FLOAT16':
                node_proto.attribute.extend([
                    make_attribute('to', TensorProto.FLOAT16)])
            if arg.s.upper() == b'FLOAT32':
                node_proto.attribute.extend([
                    make_attribute('to', TensorProto.FLOAT)])
            elif arg.s.upper() == b'FLOAT64':
                node_proto.attribute.extend([
                    make_attribute('to', TensorProto.DOUBLE)])
            else:
                node_proto.attribute.extend([
                    make_attribute('to', TensorProto.UNDEFINED)])

    return node_proto, const_tensors


def PythonONNXExporter(op_def, shape_dict, ws):
    return None, None