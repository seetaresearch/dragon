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
"""Utilities to a too simple ONNX exporting or importing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy

from dragon.core.autograph import function_lib
from dragon.core.framework import workspace
from dragon.core.proto import dragon_pb2
from dragon.core.util import serialization
from dragon.vm.onnx.frontend import graph_def_to_onnx_model
from dragon.vm.onnx.helper import mapping


def export_from_graph(
    graph_def,
    f,
    input_names=None,
    output_names=None,
    input_shapes=None,
    constants=None,
    value_info=None,
    opset_version=None,
    workspace=None,
    verbose=True,
    enable_onnx_checker=True,
):
    """Export an onnx model from the graph."""
    model = graph_def_to_onnx_model(
        graph_def=graph_def,
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes,
        constants=constants,
        value_info=value_info,
        opset_version=opset_version,
        workspace=workspace,
        verbose=verbose,
        enable_onnx_checker=enable_onnx_checker)
    serialization.save_bytes(serialization.serialize_proto(model), f)


def import_to_function(model_path, explicit_inputs=False):
    """Import an onnx model to the function."""
    return function_lib \
        .Function(name='onnx') \
        .import_from(
            graph_def=import_to_graph(model_path),
            explicit_inputs=explicit_inputs,
        )


def import_to_graph(model_path):
    """Import an onnx model to the graph."""
    if not os.path.exists(model_path):
        raise ValueError(
            'Model({}) is not existed.'
            .format(model_path)
        )
    graph_def = dragon_pb2.GraphDef()
    serialized_proto = workspace \
        .get_workspace() \
        .ImportONNXModel(model_path)
    graph_def.ParseFromString(serialized_proto)
    return graph_def


def make_value_info(shape, dtype='float32'):
    """Return a value info from the shape and data type."""
    return mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(dtype)], shape
