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

from dragon.vm.onnx.core import helper
from dragon.vm.onnx.core.exporters import utils as export_util


@export_util.register('BatchNorm')
def batch_norm_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    node.op_type = 'BatchNormalization'
    for arg in op_def.arg:
        if arg.name == 'epsilon':
            helper.add_attribute(node, 'epsilon', arg.f)
        elif arg.name == 'momentum':
            helper.add_attribute(node, 'momentum', arg.f)
        elif arg.name == 'momentum_desc':
            momentum = helper.fetch_argument(op_def, arg, context.ws)
            helper.add_attribute(node, 'momentum', float(momentum))
    return node, const_tensors


@export_util.register('ChannelNorm')
def channel_norm_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    node.op_type = 'ATen'  # Currently not supported in ai.onnx
    helper.add_attribute(node, 'op_type', 'ChannelNorm')
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
            values = helper.fetch_argument(op_def, arg, context.ws)
            helper.add_attribute(node, 'perm', values)
    return node, const_tensors


@export_util.register('GroupNorm')
def group_norm_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    node.op_type = 'ATen'  # Currently not supported in ai.onnx
    for arg in op_def.arg:
        if arg.name == 'epsilon':
            helper.add_attribute(node, 'epsilon', arg.f)
        elif arg.name == 'group':
            if arg.i == 0:
                # InstanceNorm
                node.op_type = 'InstanceNormalization'
            else:
                helper.add_attribute(node, 'op_type', 'GroupNorm')
                helper.add_attribute(node, 'group', arg.i)
    return node, const_tensors


@export_util.register('LpNorm')
def lp_norm_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    node.op_type = 'LpNormalization'
    axis, end_axis = None, None
    for arg in op_def.arg:
        if arg.name == 'axis':
            axis = arg.i
            helper.add_attribute(node, 'axis', arg.i)
        elif arg.name == 'end_axis':
            end_axis = arg.i
            if end_axis < 0:
                input_shape = context.blob_shapes[op_def.input[0]]
                end_axis += len(input_shape)
        elif arg.name == 'p':
            helper.add_attribute(node, 'p', arg.i)
    if end_axis is not None and axis != end_axis:
        raise ValueError('Reshape to avoid multiple axes.')
    return node, const_tensors
