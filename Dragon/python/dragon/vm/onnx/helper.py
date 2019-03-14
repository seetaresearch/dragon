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
#      <https://github.com/pytorch/pytorch/blob/master/caffe2/python/onnx/helper.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from onnx.backend.base import namedtupledict
from onnx import numpy_helper

import dragon as dg
from dragon.vm.onnx.workspace import Workspace


INITIALIZER_TAG = {
    'PRelu': (1,),
    'Affine': (1, 2,),
    'FullyConnected': (1, 2,),
    'BatchNorm': (1, 2, 3,),
    'BatchRenorm': (1, 2, 3,),
    'FusedBatchNorm': (1, 2, 3, 4),
    'FusedGroupNorm': (1, 2,),
    'BiasAdd': (1,),
    'Conv2d': (1, 2,),
    'ConvTranspose2d': (1, 2,),
    'DepthwiseConv2d': (1, 2,),
}


def extract_leaf_tensors(graph_def):
    collection, outputs = set(), set()
    for op in graph_def.op:
        for input in op.input:
            if input not in outputs:
                collection.add(input)
        for output in op.output:
            outputs.add(output)
    return collection


def extract_initializer(graph_def):
    collection = set()
    for op in graph_def.op:
        if op.type in INITIALIZER_TAG:
            for idx, input in enumerate(op.input):
                if idx in INITIALIZER_TAG[op.type]:
                    collection.add(input)
    return collection


def fetch_initializer(initializer):
    # Fetch the initializer
    return [
        numpy_helper.from_array(
            dg.workspace.FetchTensor(name), name=name)
                for name in initializer
    ]


def fetch_argument(op_def, desc, ws):
    if sys.version_info >= (3, 0):
        desc = desc.decode('utf-8')
    desc = desc.replace('${ANCHOR}', op_def.name)
    argument_value = ws.FetchTensor(desc)
    if argument_value.size == 1:
        return argument_value.flatten()[0]
    return argument_value


def native_run_graph(graph_def, inputs, initializer, init_func=None):
    # De-Optimization
    for i in range(len(graph_def.arg)):
        if graph_def.arg[i].name == 'optimization_level':
            graph_def.arg[i].i = 0

    # Create an anonymous workspace
    ws = Workspace()

    with dg.ws_scope(ws.name):
        # Register all the initializer before feeding them
        for name in initializer:
            dg.Tensor(name=name).Variable()

        # Feed the given values if necessary
        if init_func: init_func()

        # Feed the external inputs
        for name, blob in inputs.items():
            dg.workspace.FeedTensor(name, blob)

        # Create and Run the graph
        graph_name = dg.workspace.CreateGraph(graph_def)
        dg.workspace.RunGraph(graph_name, return_outputs=False)

        # Fetch the outputs
        output_names = graph_def.output
        output_values = [dg.workspace.FetchTensor(name) for name in output_names]

        # Fetch the initializer
        initializer = [
            numpy_helper.from_array(
                dg.workspace.FetchTensor(name), name=name)
                    for name in initializer
        ]

    # Return the outputs
    return ws, namedtupledict('Outputs', output_names)(*output_values), initializer