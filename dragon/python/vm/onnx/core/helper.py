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
"""ONNX helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy
try:
    from onnx import mapping
    from onnx.backend.base import namedtupledict
    from onnx.helper import make_attribute
    from onnx.helper import make_graph
    from onnx.helper import make_model
    from onnx.helper import make_node
    from onnx.helper import make_tensor
    from onnx.helper import make_tensor_value_info
    from onnx.helper import printable_graph
    from onnx.numpy_helper import from_array
except ImportError:
    from dragon.core.util import deprecation
    mapping = deprecation.NotInstalled('onnx')
    namedtupledict = deprecation.not_installed('onnx')
    make_attribute = deprecation.not_installed('onnx')
    make_graph = deprecation.not_installed('onnx')
    make_model = deprecation.not_installed('onnx')
    make_node = deprecation.not_installed('onnx')
    make_tensor = deprecation.not_installed('onnx')
    make_tensor_value_info = deprecation.not_installed('onnx')
    printable_graph = deprecation.not_installed('onnx')
    from_array = deprecation.not_installed('onnx')


def add_attribute(node_proto, name, value):
    """Add a new attribute into the node proto."""
    node_proto.attribute.extend([make_attribute(name, value)])


def collect_inputs(graph_def):
    """Return all input tensor names defined in the graph."""
    collection, outputs = set(), {''}
    if hasattr(graph_def, 'op'):
        ops = graph_def.op
    elif hasattr(graph_def, 'node'):
        ops = graph_def.node
    else:
        raise ValueError('GraphDef dose not have attr <op> or <node>.')
    for op in ops:
        for input in op.input:
            if input not in outputs:
                collection.add(input)
        for output in op.output:
            outputs.add(output)
    return collection


def fetch_argument(op_def, arg, ws):
    """Return the value of a tensor argument."""
    desc = arg if isinstance(arg, bytes) else arg.s
    if sys.version_info >= (3, 0):
        desc = desc.decode('utf-8')
    desc = desc.replace('$NAME', op_def.name)
    value = ws.get_tensor(desc).ToNumpy()
    if value.size == 1:
        return value.flatten()[0]
    return value


def fetch_arguments(op_def, arg, ws):
    """Return the value of tensor arguments."""
    return [fetch_argument(op_def, desc, ws) for desc in arg.strings]


def fetch_tensor(name, ws):
    """Return the value of a tensor."""
    return ws.get_tensor(name).ToNumpy()


def from_tensor(tensor, ws):
    """Return a tensor proto from the existing tensor."""
    return from_array(fetch_tensor(tensor, ws), tensor)


def make_model_from_node(node_proto, inputs, use_weights=True):
    """Make a model from the standalone node proto."""
    output_dtype = 'float32'  # Dummy value only
    output_shape = [-99]  # Dummy value only
    graph_inputs = [make_tensor_value_info(
        name, tensor_type(str(array.dtype)), array.shape)
        for name, array in zip(node_proto.input, inputs)]
    graph_outputs = [make_tensor_value_info(
        name, tensor_type(output_dtype), output_shape)
        for name in node_proto.output]
    if use_weights:
        initializers = [make_tensor(
            name, tensor_type(str(array.dtype)), array.shape,
            array.flatten().tolist())
            for name, array in zip(node_proto.input[1:], inputs[1:])]
    else:
        initializers = []
    graph = make_graph(
        [node_proto],
        "RunNodeGraph_" + node_proto.op_type,
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    model = make_model(graph)
    return model


def tensor_type(type_str):
    """Return the tensor type from a string descriptor."""
    return mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(type_str.lower())]
