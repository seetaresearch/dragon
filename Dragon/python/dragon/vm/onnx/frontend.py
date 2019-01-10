# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/pytorch/pytorch/blob/master/caffe2/python/onnx/frontend.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np

from onnx import (checker, mapping, numpy_helper, GraphProto, OperatorSetIdProto)
from onnx.helper import make_tensor_value_info, make_model, printable_graph

from dragon.vm.onnx.helper import \
    (extract_initializer, extract_leaf_tensors,
     native_run_graph,)

from dragon.vm.onnx.nodes.factory import get_nodes_def


class DragonFrontend(object):
    """This class help dragon to convert

    internal protocols to ONNX protocols.

    """
    target_opset_version = 9

    @staticmethod
    def _extract_value_info(tensor):
        return make_tensor_value_info(
            name=tensor.name,
            elem_type=tensor.data_type,
            shape=tensor.dims)

    @classmethod
    def graph_def_to_onnx_graph(
        cls,
        graph_def,
        init_func=None,
        value_info=None,
        graph_name=None,
        verbose=True
    ):
        if value_info is None: value_info = {}
        if not isinstance(value_info, dict):
            raise ValueError(
                'Please pass value_info as a '
                    'name -> (type, shape) dictionary')

        leaf_tensors = extract_leaf_tensors(graph_def)
        initializer = extract_initializer(graph_def)

        # Check whether we have got type shape info of all input
        missing = (leaf_tensors - set(value_info.keys()) - initializer)
        if missing:
            raise RuntimeError('Could not find value info of inputs: {}'.format(
                ', '.join(missing)))

        # Check if value_info contains the types/shapes of all the blobs, in
        # which case we don't need to infer them by running the net.
        run_native_graph = False
        for op in graph_def.op:
            for name in itertools.chain(op.input, op.output):
                if name not in value_info:
                    run_native_graph = True
                    break

        ws = None

        if run_native_graph:
            inputs = {}
            for name, (elem_type, shape) in value_info.items():
                inputs[name] = np.random.randn(*shape).astype(
                    mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type])

            ws, outputs, initializer = native_run_graph(
                graph_def, inputs, initializer, init_func)

            for name in graph_def.output:
                output = outputs[name]
                elem_type = mapping.NP_TYPE_TO_TENSOR_TYPE[output.dtype]
                shape = output.shape
                value_info[name] = (elem_type, shape)

        onnx_graph = GraphProto()
        onnx_graph.name = graph_name if graph_name else graph_def.name

        # Initializer should also be included in the inputs
        value_info.update({
            init.name: (init.data_type, init.dims)
                for init in initializer})

        # Add initializer
        onnx_graph.initializer.extend(initializer)

        # Add inputs
        onnx_graph.input.extend(
            make_tensor_value_info(
                name=name,
                elem_type=value_info[name][0],
                shape=value_info[name][1])
            for name in leaf_tensors)

        # Add outputs
        onnx_graph.output.extend(
            make_tensor_value_info(
                name=name,
                elem_type=value_info[name][0],
                shape=value_info[name][1])
            for name in set(graph_def.output))

        # Add nodes
        for op in graph_def.op:
            shapes = {}
            for name in itertools.chain(op.input, op.output):
                if ws:
                    blob = ws.FetchTensor(name)
                    if hasattr(blob, 'shape'):
                        shapes[name] = blob.shape
                else:
                    shapes[name] = value_info[name][1]

            # Try to translate op => nodes
            nodes, const_tensors = get_nodes_def(op, shape_dict=shapes)

            # Directly convert outputs as const tensors if necessary
            if None in nodes:
                const_tensors = [
                    numpy_helper.from_array(
                        ws.FetchTensor(name), name=name)
                            for name in op.output]
            else:
                onnx_graph.node.extend(nodes)

            # Add const tensors
            if const_tensors is not None:
                onnx_graph.initializer.extend(const_tensors)
                onnx_graph.input.extend([
                    cls._extract_value_info(tensor)
                        for tensor in const_tensors])

        if verbose: print(printable_graph(onnx_graph))

        return onnx_graph

    @classmethod
    def graph_def_to_onnx_model(
        cls,
        graph_def,
        init_func=None,
        value_info=None,
        graph_name=None,
        verbose=True
    ):
        opset_id = OperatorSetIdProto()
        opset_id.domain = ''  # ONNX default domain
        opset_id.version = cls.target_opset_version
        model = make_model(
            cls.graph_def_to_onnx_graph(
                graph_def, init_func, value_info, graph_name, verbose),
            opset_imports=[opset_id],     # current supported opset version
            producer_name='onnx-dragon',  # producer name
        )
        checker.check_model(model)
        return model

graph_def_to_onnx_graph = DragonFrontend.graph_def_to_onnx_graph
graph_def_to_onnx_model = DragonFrontend.graph_def_to_onnx_model