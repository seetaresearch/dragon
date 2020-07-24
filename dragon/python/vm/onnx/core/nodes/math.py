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

import numpy

from dragon.vm.onnx.core import exporter
from dragon.vm.onnx.core import helper


@exporter.register('Add')
def add_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    dtype = str(helper.fetch_tensor(op_def.output[0], ws).dtype)
    node.op_type = 'Or' if dtype == 'bool' else 'Add'
    const_tensors = []  # Global scalars
    for e in op_def.input:
        if e.startswith('/share/scalar/'):
            const_tensors.append(helper.from_tensor(e, ws))
    return node, const_tensors


@exporter.register('Affine')
def affine_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'ATen'  # Currently not supported in ai.onnx
    helper.add_attribute(node, 'op_type', 'Affine')
    for arg in op_def.arg:
        if arg.name == 'axis':
            helper.add_attribute(node, 'axis', arg.i)
        elif arg.name == 'num_axes':
            helper.add_attribute(node, 'num_axes', arg.i)
    # Weights and biases
    const_tensors = [helper.from_tensor(e, ws) for e in op_def.input[1:]]
    return node, const_tensors


@exporter.register('Div')
def div_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    const_tensors = []  # Global scalars
    for e in op_def.input:
        if e.startswith('/share/scalar/'):
            const_tensors.append(helper.from_tensor(e, ws))
    return node, const_tensors


@exporter.register('Clip')
def clip_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    for arg in op_def.arg:
        if arg.name == 'low':
            helper.add_attribute(node, 'min', arg.f)
        elif arg.name == 'high':
            helper.add_attribute(node, 'max', arg.f)
    return node, const_tensors


@exporter.register('Clip-11')
def clip_exporter_v11(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    min_value, max_value, const_tensors = None, None, []
    dtype = ws.FetchTensor(op_def.output[0]).dtype
    for arg in op_def.arg:
        if arg.name == 'low':
            min_value = arg.f
        elif arg.name == 'high':
            max_value = arg.f
    if min_value is not None:
        const_tensors.append(helper.from_array(
            numpy.array(min_value, dtype),
            op_def.input[0] + '/clip/min_value',
        ))
        node.input.extend([const_tensors[-1].name])
    else:
        node.input.extend([''])
    if max_value is not None:
        const_tensors.append(helper.from_array(
            numpy.array(max_value, dtype),
            op_def.input[0] + '/clip/max_value',
        ))
        node.input.extend([const_tensors[-1].name])
    else:
        node.input.extend([''])
    return node, const_tensors


@exporter.register('FullyConnected-7')
def fully_connected_exporter_v7(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'Gemm'
    helper.add_attribute(node, 'alpha', 1.)
    helper.add_attribute(node, 'beta', 1.)
    for arg in op_def.arg:
        if arg.name == 'transW':
            helper.add_attribute(node, 'transB', arg.i)
    # Weights and biases
    const_tensors = [helper.from_tensor(e, ws) for e in op_def.input[1:]]
    return node, const_tensors


@exporter.register('FullyConnected')
def fully_connected_exporter(op_def, shape_dict, ws):
    node, const_tensors = fully_connected_exporter_v7(op_def, shape_dict, ws)
    helper.add_attribute(node, 'broadcast', 1)  # Removed since opset 7
    return node, const_tensors


@exporter.register('Invert')
def invert_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'Not'
    return node, const_tensors


@exporter.register('Matmul')
def matmul_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'MatMul'
    for arg in op_def.arg:
        if arg.name == 'transA':
            if arg.i > 0:
                raise ValueError('Matmul requires an non-transposed matrix a.')
        elif arg.name == 'transB':
            if arg.i > 0:
                raise ValueError('Matmul requires an non-transposed matrix b.')
    return node, const_tensors


@exporter.register('Maximum')
def maximum_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'Max'  # Eltwise, Broadcast
    const_tensors = []  # Global scalars
    for e in op_def.input:
        if e.startswith('/share/scalar/'):
            const_tensors.append(helper.from_tensor(e, ws))
    return node, const_tensors


@exporter.register('Minimum')
def minimum_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    node.op_type = 'Min'  # Eltwise, Broadcast
    const_tensors = []  # Global scalars
    for e in op_def.input:
        if e.startswith('/share/scalar/'):
            const_tensors.append(helper.from_tensor(e, ws))
    return node, const_tensors


@exporter.register('Mul')
def mul_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    dtype = str(helper.fetch_tensor(op_def.output[0], ws).dtype)
    node.op_type = 'And' if dtype == 'bool' else 'Mul'
    const_tensors = []  # Global scalars
    for e in op_def.input:
        if e.startswith('/share/scalar/'):
            const_tensors.append(helper.from_tensor(e, ws))
    return node, const_tensors


@exporter.register('Pow')
def pow_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    const_tensors = []  # Global scalars
    for e in op_def.input:
        if e.startswith('/share/scalar/'):
            const_tensors.append(helper.from_tensor(e, ws))
    return node, const_tensors


@exporter.register('Sub')
def sub_exporter(op_def, shape_dict, ws):
    node, const_tensors = exporter.translate(**locals())
    dtype = str(helper.fetch_tensor(op_def.output[0], ws).dtype)
    node.op_type = 'Xor' if dtype == 'bool' else 'Sub'
    const_tensors = []  # Global scalars
    for e in op_def.input:
        if e.startswith('/share/scalar/'):
            const_tensors.append(helper.from_tensor(e, ws))
    return node, const_tensors
