# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.onnx import exporter
from dragon.vm.onnx import helper


@exporter.register('BatchNorm')
def batch_norm_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'BatchNormalization'

    for arg in op_def.arg:
        if arg.name == 'eps':
            helper.add_attribute(node, 'epsilon', arg.f)
        elif arg.name == 'momentum':
            helper.add_attribute(node, 'momentum', arg.f)

    # Weight, bias, running mean and running variance
    const_tensors = [helper.from_tensor(e, ws) for e in op_def.input[1:]]

    return node, const_tensors


@exporter.register('GroupNorm')
def group_norm_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'ATen'  # Currently not supported in ai.onnx

    for arg in op_def.arg:
        if arg.name == 'eps':
            helper.add_attribute(node, 'epsilon', arg.f)
        elif arg.name == 'group':
            if arg.i == 0:
                # InstanceNorm
                node.op_type = 'InstanceNormalization'
            else:
                helper.add_attribute(node, 'op_type', 'GroupNorm')
                helper.add_attribute(node, 'group', arg.i)

    # Weight and bias
    const_tensors = [helper.from_tensor(e, ws) for e in op_def.input[1:]]

    return node, const_tensors


@exporter.register('LpNormalize')
def lp_normalize_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'LpNormalization'

    for arg in op_def.arg:
        if arg.name == 'axis':
            helper.add_attribute(node, 'axis', arg.i)
        if arg.name == 'p':
            helper.add_attribute(node, 'p', arg.i)

    return node, const_tensors
