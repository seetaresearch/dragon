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

import numpy as np
from onnx import numpy_helper
from onnx.helper import make_attribute
from dragon.vm.onnx.nodes.common import CommonONNXExporter


def _normalize_tuple(value, rank):
    if len(value) > rank:
        return [value[i] for i in range(rank)]
    else:
        return [value[i] for i in range(len(value))] + \
            [value[-1] for i in range(len(value), rank)]


def ReshapeONNXExporter(op_def, shape_dict):
    output_shape = list(shape_dict[op_def.output[0]])
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    for arg in op_def.arg:
        if arg.name == 'dims':
            for axis, s in enumerate(arg.ints):
                if s == -1 or s == 0:
                    output_shape[axis] = s
                else:
                    if s != output_shape[axis]:
                        raise ValueError('Expected shape[{}] to be {}, but go {}.\n'
                            'Please follow the static data shape on exporting.'.format(
                                axis, s, output_shape[axis]))

    shape = numpy_helper.from_array(
        np.array(output_shape, dtype=np.int64),
            name=op_def.input[0] + '/onnx/shape')
    node_proto.input.extend([shape.name])

    return node_proto, [shape]


def ConcatONNXExporter(op_def, shape_dict):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    for arg in op_def.arg:
        if arg.name == 'axis':
            node_proto.attribute.extend([
                make_attribute('axis', arg.i)])

    return node_proto, None


def FlattenONNXExporter(op_def, shape_dict):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    for arg in op_def.arg:
        if arg.name == 'axis':
            node_proto.attribute.extend([
                make_attribute('axis', arg.i)])
        elif arg.name == 'num_axes':
            if arg.i != -1:
                raise ValueError('Excepted num_axes == -1, but got {}.'.format(arg.i))
        elif arg.name == 'keep_axes':
            raise ValueError('keep_axes should not be set. (Theano Style).')

    return node_proto, None


def TransposeONNXExporter(op_def, shape_dict):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    for arg in op_def.arg:
        if arg.name == 'perm':
            node_proto.attribute.extend([
                make_attribute('perm', arg.ints)])

    return node_proto, None


def ArgReduceONNXExporter(op_def, shape_dict):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)

    # ONNX requires indices only, remove the values
    indices = node_proto.output[0]
    node_proto.ClearField('output')
    node_proto.output.extend([indices])

    for arg in op_def.arg:
        if arg.name == 'axis':
            node_proto.attribute.extend([
                make_attribute('axis', arg.i)])
        elif arg.name == 'keep_dims':
            node_proto.attribute.extend([
                make_attribute('keepdims', arg.i)])
        elif arg.name == 'top_k':
            if arg.i != 1:
                raise ValueError('ONNX requires top_k == 1.')
        elif arg.name == 'operation':
            if arg.s == b'ARGMAX':
                node_proto.op_type = 'ArgMax'
            elif arg.s == b'ARGMIN':
                node_proto.op_type = 'ArgMin'

    return node_proto, None


def CropONNXExporter(op_def, shape_dict):
    node_proto, const_tensors = CommonONNXExporter(op_def, shape_dict)
    node_proto.op_type = 'ATen' # Template
    node_proto.attribute.extend([make_attribute('op_type', 'Crop')])

    for arg in op_def.arg:
        if arg.name == 'starts':
            if len(arg.ints) > 0:
                node_proto.attribute.extend([
                    make_attribute('starts', arg.ints)])
        elif arg.name == 'ends':
            if len(arg.ints) > 0:
                node_proto.attribute.extend([
                    make_attribute('ends', arg.ints)])
        elif arg.name == 'start_axis':
            node_proto.attribute.extend([
                make_attribute('start_axis', arg.i)])
        elif arg.name == 'offsets':
            node_proto.attribute.extend([
                make_attribute('offsets', arg.ints)])
        elif arg.name == 'shape':
            node_proto.attribute.extend([
                make_attribute('shape', arg.ints)])
        elif arg.name == 'shape_like':
            node_proto.attribute.extend([
                make_attribute('shape_like', arg.s)])

    return node_proto, None