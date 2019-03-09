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

"""Utilities to a too simple ONNX exporting or importing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from onnx import mapping
from google.protobuf.text_format import Parse as parse_text_proto

import dragon.proto.dragon_pb2 as pb
import dragon.import_c_api as C

from dragon.vm.theano.compile.function import Function
from dragon.vm.onnx.frontend import graph_def_to_onnx_model
from dragon.vm.onnx.serialization import save_model


def export_from_graph_def(
    graph_def,
    f,
    init_func=None,
    constants=None,
    value_info=None,
    graph_name=None,
    verbose=True,
    enforce_no_running=False,
):
    """Export a ONNX model from the ``GraphDef``.

    Set the ``init_func`` to process the initializer before running.

    ``value_info`` should be set as (name, (onnx.TensorProto.DTYPE, shape)).

    Parameters
    ----------
    graph_def : GraphDef
        The specific graph def.
    f : str
        The exporting model path.
    init_func : lambda, optional
        The init function to execute before native running.
    constants : dict, optional
        The value of external const tensors.
    value_info : dict, optional
        The info external const tensors.
    graph_name : str, optional
        The optional graph name.
    verbose : boolean, optional
        Whether to print the ONNX graph.
    enforce_no_running : boolean, optional
        Whether not to run this graph def.

    Returns
    -------
    onnx_pb2.ModelProto
        The ONNX model.

    """
    save_model(graph_def_to_onnx_model(
        graph_def=graph_def,
        init_func=init_func,
        constants=constants,
        value_info=value_info,
        graph_name=graph_name,
        verbose=verbose,
        enforce_no_running=enforce_no_running,
    ), f)


def export_from_graph_text(
    text_file,
    f,
    init_func=None,
    constants=None,
    value_info=None,
    graph_name=None,
    verbose=True,
):
    """Export a ONNX model from the textified ``GraphDef``.

    Set the ``init_func`` to process the initializer before running.

    ``value_info`` should be set as (name, (onnx.TensorProto.DTYPE, shape)).

    Parameters
    ----------
    text_file : str
        The path to the text file of graph def.
    f : str
        The file descriptor.
    init_func : lambda, optional
        The init function to execute before native running.
    constants : dict, optional
        The value of external const tensors.
    value_info : dict, optional
        The info external const tensors.
    graph_name : str, optional
        The optional graph name.
    verbose : bool, optional
        Whether to print the ONNX graph.

    Returns
    -------
    onnx_pb2.ModelProto
        The ONNX model.

    """
    with open(text_file, 'r') as rf:
        graph_def = pb.GraphDef()
        parse_text_proto(rf.read(), graph_def)

    export_from_graph_def(
        graph_def=graph_def,
        f=f,
        init_func=init_func,
        constants=constants,
        value_info=value_info,
        graph_name=graph_name,
        verbose=verbose)


def import_to_graph_def(model_path):
    """Import a ONNX model to the graph def.

    Parameters
    ----------
    model_path : str
        The path to the ONNX model.

    Returns
    -------
    GraphDef
        The translated graph def.

    """
    if not os.path.exists(model_path):
        raise ValueError('Given model({}) is not existed.'.format(model_path))
    graph_def = pb.GraphDef()
    serialized_proto = C.ImportONNXModel(model_path)
    graph_def.ParseFromString(serialized_proto)
    return graph_def


def import_to_function(model_path, explicit_inputs=False):
    """Import a ONNX model to the Function.

    Parameters
    ----------
    model_path : str
        The path to the ONNX model.
    explicit_inputs : boolean
        Whether to attach the external inputs to the function.

    Returns
    -------
    Function
        The translated function.

    """
    return Function(name='onnx').import_from(
        import_to_graph_def(model_path), explicit_inputs)


def surgery_on_graph_def(
    graph_def,
    renamed_tensors={},
    external_inputs=(),
    external_outputs=(),
):
    """Perform a surgery on a graph def for easy exporting.

    Set ``external_inputs`` or ``external_outputs`` to override
    the original settings from the Function.

    Then, set the ``renamed_tensors`` to replace the existing tensors
    (including ``external_inputs`` and ``external_outputs``) with new tensors.

    Parameters
    ----------
    graph_def : GraphDef
        The specific graph def.
    renamed_tensors : dict of (string, Tensor)
        The dict to store the renames.
    external_inputs : tuple, list of (string, Tensor)
        The override external inputs.
    external_outputs : tuple, list of (string, Tensor)
        The override external outputs.

    Returns
    -------
    graph_def : GraphDef
        The modified graph def.

    """
    inputs = [e.name if hasattr(e, 'name') else e for e in external_inputs]
    outputs = [e.name if hasattr(e, 'name') else e for e in external_outputs]

    if len(inputs) > 0:
        graph_def.ClearField('input')
        graph_def.input.extend(inputs)

    if len(outputs) > 0:
        graph_def.ClearField('output')
        graph_def.output.extend(outputs)

    renamed_dict = dict()

    for k, v in renamed_tensors.items():
        renamed_dict[k.name if hasattr(k, 'name') else k] \
            = v.name if hasattr(v, 'name') else v

    def renamed_func(repeated_message):
        for idx, e in enumerate(repeated_message):
            if e in renamed_dict:
                repeated_message[idx] = renamed_dict[e]

    renamed_func(graph_def.input)
    renamed_func(graph_def.output)

    for op in graph_def.op:
        renamed_func(op.input)
        renamed_func(op.output)

    return graph_def


def make_value_info(shape, dtype='float32'):
    return mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], shape