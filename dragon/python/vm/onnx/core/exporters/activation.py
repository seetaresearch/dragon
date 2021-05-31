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


@export_util.register('Dropout')
def dropout_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    drop_ratio = 0.5  # The prob to set zeros randomly.
    for arg in op_def.arg:
        if arg.name == 'prob':
            drop_ratio = arg.f
        elif arg.name == 'prob_desc':
            drop_ratio = helper.fetch_argument(op_def, arg, context.ws)
    helper.add_attribute(node, 'ratio', float(drop_ratio))
    return node, const_tensors


@export_util.register('HardSigmoid')
def hardsigmoid_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    alpha, beta = 0.2, 0.5
    for arg in op_def.arg:
        if arg.name == 'alpha':
            alpha = arg.f
        elif arg.name == 'beta':
            beta = arg.f
    helper.add_attribute(node, 'alpha', alpha)
    helper.add_attribute(node, 'beta', beta)
    return node, const_tensors


@export_util.register('Relu')
def relu_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    for arg in op_def.arg:
        if arg.name == 'alpha':
            if arg.f > 0:
                node.op_type = 'LeakyRelu'
                helper.add_attribute(node, 'alpha', arg.f)
    return node, const_tensors


@export_util.register('Selu')
def selu_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    alpha, gamma = 1.67326, 1.0507
    for arg in op_def.arg:
        if arg.name == 'alpha':
            alpha = arg.f
        elif arg.name == 'gamma':
            gamma = arg.f
    helper.add_attribute(node, 'alpha', alpha)
    helper.add_attribute(node, 'gamma', gamma)
    return node, const_tensors


@export_util.register(['Softmax', 'LogSoftmax'])
def softmax_exporter(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    ndim = len(context.blob_shapes[op_def.input[0]])
    for arg in op_def.arg:
        if arg.name == 'axis':
            axis = arg.i + (ndim if arg.i < 0 else 0)
            if axis != (ndim - 1):
                raise ValueError('Axis could only be the last if opset < 13.')
            helper.add_attribute(node, 'axis', arg.i)
    return node, const_tensors


@export_util.register(['Softmax-13', 'LogSoftmax-13'])
def softmax_exporter_v13(op_def, context):
    node, const_tensors = export_util.translate(**locals())
    ndim = len(context.blob_shapes[op_def.input[0]])
    for arg in op_def.arg:
        if arg.name == 'axis':
            axis = arg.i + (ndim if arg.i < 0 else 0)
            helper.add_attribute(node, 'axis', arg.i)
    return node, const_tensors
