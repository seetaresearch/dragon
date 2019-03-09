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

from onnx.helper import make_attribute
from dragon.vm.onnx.nodes.common import CommonONNXExporter


def _assert_data_format(arg):
    if arg.name == 'axis':
        if arg.i != 1:
            raise ValueError('ONNX only support NCHW format.')


def BatchNormONNXExporter(op_def, shape_dict, ws):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    # [x, mean, var, scale, bias] => [x, scale, bias, mean, var]
    mean, var, scale, bias = node_proto.input[1:5]
    node_proto.input[1:5] = scale, bias, mean, var

    node_proto.attribute.extend([
        make_attribute('spatial', 1)])

    for arg in op_def.arg:
        if arg.name == 'eps':
            node_proto.attribute.extend([
                make_attribute('epsilon', arg.f)])
        elif arg.name == 'momentum':
            node_proto.attribute.extend([
                make_attribute('momentum', arg.f)])

    return node_proto, const_tensors


def L2NormONNXExporter(op_def, shape_dict, ws):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    node_proto.attribute.extend([
        make_attribute('p', 2)])

    for arg in op_def.arg:
        if arg.name == 'axis':
            node_proto.attribute.extend([
                make_attribute('axis', arg.i)])

    return node_proto, const_tensors