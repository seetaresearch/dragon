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


def SoftmaxONNXExporter(op_def, shape_dict):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    for arg in op_def.arg:
        if arg.name == 'axis':
            node_proto.attribute.extend([
                make_attribute('axis', int(arg.i))])

    return node_proto, const_tensors