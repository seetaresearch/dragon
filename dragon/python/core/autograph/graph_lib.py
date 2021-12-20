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
"""Graph library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools
import weakref

from dragon.core import distributed
from dragon.core.framework import backend
from dragon.core.framework import config
from dragon.core.framework import context
from dragon.core.framework import proto_util
from dragon.core.framework import tapes
from dragon.core.framework import workspace
from dragon.core.proto import dragon_pb2
from dragon.core.util import nest


class GraphExec(object):
    """The executable graph."""

    def __init__(self, graph_def, execute_ws):
        self._def = graph_def
        self._workspace_ref = weakref.ref(execute_ws)

    def run(self):
        """Run graph in the created workspace."""
        self._workspace_ref().run_graph(self._def.name)


class GraphLib(object):
    """Library class to create the graph from various targets."""

    @staticmethod
    def from_onnx(model, name=None):
        """Create a graph from the onnx model."""
        execute_ws = workspace.get_workspace()
        graph_str = execute_ws._impl.PrepareONNXModel(model)
        graph_def = dragon_pb2.GraphDef()
        graph_def.ParseFromString(graph_str)
        graph_def.name = 'Graph' if name is None else name
        GraphLib._add_device(graph_def)
        GraphLib._add_optimization(graph_def)
        for input in graph_def.input:
            execute_ws.create_tensor(input)
        graph_def.name = execute_ws.create_graph(graph_def)
        return GraphExec(graph_def, execute_ws)

    @staticmethod
    def from_outputs(outputs, name=None):
        """Create a graph from the output tensors."""
        outputs = nest.flatten(outputs)
        name = 'Graph' if name is None else name
        execute_ws = workspace.get_workspace()
        graph_def = dragon_pb2.GraphDef(name=name)
        GraphLib._add_outputs(graph_def, outputs)
        GraphLib._add_grads(graph_def, outputs)
        GraphLib._add_device(graph_def)
        GraphLib._add_optimization(graph_def)
        graph_def.name = execute_ws.create_graph(graph_def)
        return GraphExec(graph_def, execute_ws)

    @staticmethod
    def from_updates(grads_and_vars, optimizer, name=None):
        """Create a graph from the updates."""
        name = 'Graph' if name is None else name
        execute_ws = workspace.get_workspace()
        graph_def = dragon_pb2.GraphDef(name=name)
        GraphLib._add_updates(graph_def, grads_and_vars, optimizer)
        GraphLib._add_device(graph_def)
        graph_def.name = execute_ws.create_graph(graph_def)
        return GraphExec(graph_def, execute_ws)

    @staticmethod
    def _add_outputs(graph_def, outputs):
        """Add outputs."""
        op_tape = tapes.OrderedTape()
        inputs = list(outputs)
        while len(inputs) > 0:
            input = inputs.pop(0)
            if input._tape:
                op_tape.merge_from(input._tape)
                inputs.extend(input._tape.get_sources())
                input._tape = None
        graph_def.op.extend(copy.deepcopy(op_tape.get_elements()))
        graph_def.output.extend([output.id for output in outputs])
        graph_def.output.extend(op_tape.get_targets())

    @staticmethod
    def _add_device(graph_def):
        """Add device."""
        cfg = config.config()
        spec = context.get_device()
        graph_def.device_option.CopyFrom(
            proto_util.get_device_option(
                spec.type, spec.index, cfg.random_seed))

    @staticmethod
    def _add_grads(graph_def, outputs):
        """Add gradients."""
        grad_tape = tapes.Tape()
        grad_outputs = []
        for i, output in enumerate(outputs):
            if hasattr(output, '_grad_tape') and output._grad_tape:
                if output._grad_tape != grad_tape and len(grad_outputs) > 0:
                    raise RuntimeError('Create graph from multiple gradient tapes.')
                grad_tape = output._grad_tape
                output._grad_tape = None
                grad_outputs.append(output)
        if grad_tape is None:
            return
        op_defs = grad_tape.get_elements()
        if len(op_defs) == 0:
            return
        execute_ws = workspace.get_workspace()
        ys = [y.id for y in grad_outputs]
        dys = [getattr(y._grad, 'id', '') for y in grad_outputs]
        grad_defs = backend.GradientTape().CreateGradientDefs(
            [op_def.SerializeToString() for op_def in op_defs], ys, dys)
        for serialized_str in grad_defs:
            grad_def = dragon_pb2.OperatorDef()
            grad_def.ParseFromString(serialized_str)
            grad_def.name = execute_ws.create_handle('Op')
            graph_def.op.extend([grad_def])
        if len(grad_defs) > 0:
            xs = [x.id for x in grad_tape.get_sources()]
            graph_def.arg.extend([
                proto_util.make_argument('grad_sources', xs),
                proto_util.make_argument('phase', 'TRAIN')])

    @staticmethod
    def _add_optimization(graph_def, level=None):
        """Add optimization info."""
        cfg = config.config()
        if level is None:
            level = cfg.graph_optimization
        graph_def.arg.add().CopyFrom(
            proto_util.make_argument('optimization', level))
        graph_def.type = cfg.graph_type

    @staticmethod
    def _add_updates(graph_def, grads_and_vars, optimizer):
        group_vars = collections.defaultdict(list)
        group_grads = collections.defaultdict(list)
        for grad, var in grads_and_vars:
            weight_decay = getattr(var, '_weight_decay', None)
            if weight_decay is not None:
                weight_decay = float(weight_decay)
            group_vars[weight_decay].append(var.id)
            group_grads[weight_decay].append(grad.id)
        op_defs = []
        process_group = distributed.get_group()
        if process_group:
            grads = list(itertools.chain(*group_grads.values()))
            op_defs.append(proto_util.make_operator_def(
                op_type='Collective',
                inputs=grads,
                outputs=grads,
                name=optimizer._name,
                operation='ALLREDUCE',
                reduction='MEAN',
                **process_group.arguments))
        for weight_decay, vars in group_vars.items():
            grads = group_grads[weight_decay]
            op_defs.append(proto_util.make_operator_def(
                op_type=optimizer._op_type,
                inputs=grads,
                outputs=vars,
                name=optimizer._name,
                weight_decay=weight_decay))
        graph_def.op.extend(op_defs)
