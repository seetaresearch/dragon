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


def ProposalONNXExporter(op_def, shape_dict, ws):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)
    node_proto.op_type = 'ATen' # Template
    node_proto.attribute.extend([make_attribute('op_type', 'Proposal')])

    for arg in op_def.arg:
        if arg.name == 'strides':
            node_proto.attribute.extend([
                make_attribute('strides', arg.ints)])
        elif arg.name == 'ratios':
            node_proto.attribute.extend([
                make_attribute('ratios', arg.floats)])
        elif arg.name == 'scales':
            node_proto.attribute.extend([
                make_attribute('scales', arg.floats)])
        elif arg.name == 'pre_nms_top_n':
            node_proto.attribute.extend([
                make_attribute('pre_nms_top_n', arg.i)])
        elif arg.name == 'post_nms_top_n':
            node_proto.attribute.extend([
                make_attribute('post_nms_top_n', arg.i)])
        elif arg.name == 'nms_thresh':
            node_proto.attribute.extend([
                make_attribute('nms_thresh', arg.f)])
        elif arg.name == 'min_size':
            node_proto.attribute.extend([
                make_attribute('min_size', arg.i)])
        elif arg.name == 'min_level':
            node_proto.attribute.extend([
                make_attribute('min_level', arg.i)])
        elif arg.name == 'max_level':
            node_proto.attribute.extend([
                make_attribute('max_level', arg.i)])
        elif arg.name == 'canonical_scale':
            node_proto.attribute.extend([
                make_attribute('canonical_scale', arg.i)])
        elif arg.name == 'canonical_level':
            node_proto.attribute.extend([
                make_attribute('canonical_level', arg.i)])

    return node_proto, const_tensors