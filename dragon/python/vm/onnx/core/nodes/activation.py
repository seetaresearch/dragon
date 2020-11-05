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

from dragon.vm.onnx.core import exporter
from dragon.vm.onnx.core import helper


@exporter.register('Dropout')
def dropout_exporter(op_def, shape_dict, ws):
    """
    Return an exporter from ) dropout operations.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    drop_ratio = 0.5  # The prob to set zeros randomly.
    for arg in op_def.arg:
        if arg.name == 'prob':
            drop_ratio = arg.f
        elif arg.name == 'prob_desc':
            drop_ratio = helper.fetch_argument(op_def, arg, ws)
    helper.add_attribute(node, 'ratio', drop_ratio)
    return node, const_tensors


@exporter.register('HardSigmoid')
def hardsigmoid_exporter(op_def, shape_dict, ws):
    """
    Hardsigmoid for hards.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    alpha, beta = 0.2, 0.5
    for arg in op_def.arg:
        if arg.name == 'alpha':
            alpha = arg.f
        elif arg.name == 'beta':
            beta = arg.f
    helper.add_attribute(node, 'alpha', alpha)
    helper.add_attribute(node, 'beta', beta)
    return node, const_tensors


@exporter.register('PRelu')
def prelu_exporter(op_def, shape_dict, ws):
    """
    Build a tensor.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    const_tensors = [helper.from_tensor(op_def.input[1], ws)]
    return node, const_tensors


@exporter.register('Relu')
def relu_exporter(op_def, shape_dict, ws):
    """
    Return an op_def.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    for arg in op_def.arg:
        if arg.name == 'alpha':
            if arg.f > 0:
                node.op_type = 'LeakyRelu'
                helper.add_attribute(node, 'alpha', arg.f)
    return node, const_tensors


@exporter.register('Selu')
def selu_exporter(op_def, shape_dict, ws):
    """
    Return an exporter operator.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    alpha, gamma = 1.67326, 1.0507
    for arg in op_def.arg:
        if arg.name == 'alpha':
            alpha = arg.f
        elif arg.name == 'gamma':
            gamma = arg.f
    helper.add_attribute(node, 'alpha', alpha)
    helper.add_attribute(node, 'gamma', gamma)
    return node, const_tensors


@exporter.register('Softmax')
def softmax_exporter(op_def, shape_dict, ws):
    """
    Translate softmax.

    Args:
        op_def: (todo): write your description
        shape_dict: (dict): write your description
        ws: (todo): write your description
    """
    node, const_tensors = exporter.translate(**locals())
    ndim = len(shape_dict[op_def.input[0]])
    for arg in op_def.arg:
        if arg.name == 'axis':
            axis = arg.i + (ndim if arg.i < 0 else 0)
            if axis != (ndim - 1):
                raise ValueError(
                    'Softmax axis could only be the last one.\n'
                    'Use Exp(LogSoftmax) to compute the softmax instead.')
            helper.add_attribute(node, 'axis', arg.i)
    return node, const_tensors
