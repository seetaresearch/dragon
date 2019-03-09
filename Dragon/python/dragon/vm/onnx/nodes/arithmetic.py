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


def GemmONNXExporter(op_def, shape_dict, ws):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    node_proto.attribute.extend([
        make_attribute('alpha', 1.0),
            make_attribute('beta', 1.0)])
    for arg in op_def.arg:
        if arg.name == 'transW':
            node_proto.attribute.extend([
                make_attribute('transB', arg.i)])

    return node_proto, const_tensors


def AffineONNXExporter(op_def, shape_dict, ws):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)
    node_proto.op_type = 'ATen' # Template
    node_proto.attribute.extend([make_attribute('op_type', 'Affine')])

    for arg in op_def.arg:
        if arg.name == 'axis':
            node_proto.attribute.extend([
                make_attribute('axis', arg.i)])
        elif arg.name == 'num_axes':
            node_proto.attribute.extend([
                make_attribute('num_axes', arg.i)])

    return node_proto, const_tensors