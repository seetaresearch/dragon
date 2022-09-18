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
"""Native ONNX frontend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import numpy
try:
    import onnx
except ImportError:
    onnx = None
from packaging.version import parse as version_parse

from dragon.core.autograph import context as eager_context
from dragon.core.autograph.graph_lib import GraphLib
from dragon.core.framework import tapes
from dragon.core.framework import types
from dragon.core.framework import workspace as workspace_util
from dragon.core.proto import dragon_pb2
from dragon.core.util import nest
from dragon.core.util import serialization
from dragon.vm.onnx.core import helper
from dragon.vm.onnx.core.exporters import utils as export_util


class DragonFrontend(object):
    """Convert the format of IR from dragon to onnx."""

    OPSET_VERSIONS = collections.OrderedDict([
        (1, '1.0'),
        (5, '1.1'),
        (6, '1.1.2'),
        (7, '1.2'),
        (8, '1.3'),
        (9, '1.4.1'),
        (10, '1.5.0'),
        (11, '1.6.0'),
        (12, '1.7.0'),
        (13, '1.8.0'),
        (14, '1.9.0'),
        (15, '1.10.0'),
        (16, '1.11.0'),
        (17, '1.12.0'),
    ])

    @classmethod
    def graph_def_to_onnx_graph(
        cls,
        graph_def,
        input_names=None,
        output_names=None,
        input_shapes=None,
        constants=None,
        value_info=None,
        opset_version=None,
        workspace=None,
        verbose=True,
    ):
        input_names = [] if input_names is None else input_names
        output_names = [] if output_names is None else output_names
        constants = {} if constants is None else constants
        value_info = {} if value_info is None else value_info

        if not nest.is_sequence(input_names):
            raise ValueError('<input_names> should be a sequence.')
        if not nest.is_sequence(output_names):
            raise ValueError('<output_names> should be a sequence.')
        if not isinstance(constants, dict):
            raise ValueError('<constants> should be a dict with name -> value.')
        if not isinstance(value_info, dict):
            raise ValueError('<value_info> should be a dict with name -> (dtype, shape).')

        # Determine the opset version to select exporters.
        if opset_version is None:
            opset_version = cls._check_opset_version(opset_version)

        # Create aliases for blobs.
        blob_aliases = {}
        for i, alias in enumerate(output_names):
            blob_aliases[graph_def.output[i]] = alias
            if graph_def.output[i] in value_info:
                value_info[alias] = value_info[graph_def.output[i]]
        for i, alias in enumerate(input_names):
            blob_aliases[graph_def.input[i]] = alias
            if graph_def.input[i] in value_info:
                value_info[alias] = value_info[graph_def.input[i]]

        # Maybe rewrite the input shapes for future development.
        # A common case is that we should fill ``-1`` for dynamic dimension
        # in the inference runtime like TensorRT.
        if input_shapes is not None:
            if isinstance(input_shapes, dict):
                for k, v in input_shapes.items():
                    value_info[k] = (value_info[k][0], v)
            else:
                for k, v in zip(graph_def.input[:], input_shapes):
                    value_info[k] = (value_info[k][0], v)

        # Prepare to make the graph.
        onnx_graph = onnx.GraphProto(
            name=graph_def.name if len(graph_def.name) > 0 else 'onnx-model')
        blob_shapes, blob_names = {}, {}
        blob_versions = collections.defaultdict(
            int, **dict((blob_aliases.get(k, k), 1)
                        for k in helper.collect_inputs(graph_def)))
        initializers, seen_initializers = [], set()

        # Build translator context.
        context = export_util.TranslatorContext(
            workspace=workspace,
            blob_names=blob_names,
            blob_shapes=blob_shapes,
            blob_versions=blob_versions,
            opset_version=opset_version,
        )

        # Add nodes.
        for op in graph_def.op:
            # Get the shape of inputs and outputs.
            for name in itertools.chain(op.input, op.output):
                impl = workspace.get_tensor(name)
                if impl is not None:
                    blob_shapes[name] = impl.dims
                else:
                    blob_shapes[name] = value_info[name][1]

            # Translate definition.
            nodes, const_tensors = cls._make_node(op, context)

            # Rewritten for names.
            for node in nodes:
                node.input[:] = [blob_aliases.get(x, x) for x in node.input]
                node.output[:] = [blob_aliases.get(y, y) for y in node.output]
                cls._rewrite_for_ssa(node, context)

            # Convert constant outputs if necessary.
            if None in nodes:
                const_tensors = [helper.from_tensor(name, workspace)
                                 for name in op.output]
            else:
                onnx_graph.node.extend(nodes)

            # Merge constant tensors.
            if const_tensors is not None:
                value_info = {**value_info,
                              **dict((e.name, (e.data_type, e.dims))
                                     for e in const_tensors)}
                for tensor in const_tensors:
                    if tensor.name not in seen_initializers:
                        initializers.append(tensor)
                        seen_initializers.add(tensor.name)

        # Add constants.
        if constants is not None:
            for k, v in constants.items():
                initializers.append(helper.from_array(v, name=k))

        # Add inputs.
        for name in helper.collect_inputs(onnx_graph):
            try:
                onnx_graph.input.extend([
                    helper.make_tensor_value_info(
                        name=name,
                        elem_type=value_info[name][0],
                        shape=value_info[name][1])])
            except KeyError:
                impl = workspace.get_tensor(name)
                if impl is not None:
                    initializer = helper.from_tensor(name, workspace)
                    onnx_graph.input.extend([
                        helper.make_tensor_value_info(
                            name=name,
                            elem_type=initializer.data_type,
                            shape=initializer.dims)])
                    if name not in seen_initializers:
                        initializers.append(initializer)
                        seen_initializers.add(initializer.name)
                else:
                    raise ValueError(
                        'Info of tensor `{}` is missing, '
                        'specify it in <value_info>.'.format(name))

        # Add initializers.
        onnx_graph.initializer.extend(initializers)

        # Add outputs.
        onnx_graph.output.extend(
            helper.make_tensor_value_info(
                name=blob_names.get(name_v2, name_v2),
                elem_type=value_info[name_v2][0],
                shape=value_info[name_v2][1])
            for name_v2 in [blob_aliases.get(name, name)
                            for name in set(graph_def.output)])

        if verbose:
            print(helper.printable_graph(onnx_graph))

        return onnx_graph

    @classmethod
    def graph_def_to_onnx_model(
        cls,
        graph_def,
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
        opset_id = onnx.OperatorSetIdProto()
        opset_id.domain = ''  # ONNX default domain
        opset_id.version = cls._check_opset_version(opset_version)
        model = helper.make_model(
            cls.graph_def_to_onnx_graph(
                graph_def,
                input_names,
                output_names,
                input_shapes,
                constants,
                value_info,
                opset_id.version,
                workspace,
                verbose,
            ),
            opset_imports=[opset_id],  # Current supported opset version
            producer_name='onnx-dragon',  # Producer name
        )
        if enable_onnx_checker:
            onnx.checker.check_model(model)
        return model

    @classmethod
    def _check_opset_version(cls, opset_version):
        if opset_version is None:
            opset_version = list(cls.OPSET_VERSIONS.keys())[-1]
        else:
            if opset_version not in cls.OPSET_VERSIONS:
                detail_msg = 'OpSet %d is not supported.\n' % opset_version
                detail_msg += 'Following opset versions are available: {\n'
                for k, v in cls.OPSET_VERSIONS.items():
                    detail_msg += '  * Opset = %d, ONNX >= %s,\n' % (k, v)
                raise ValueError(detail_msg + '}')
        onnx_version = cls.OPSET_VERSIONS[opset_version]
        if version_parse(onnx.__version__) < version_parse(onnx_version):
            raise RuntimeError(
                'OpSet {} requires ONNX version >= {} '
                '({} currently installed.)'
                .format(opset_version, onnx_version, onnx.__version__))
        return opset_version

    @staticmethod
    def _rewrite_for_ssa(op_def, context):
        """Rewrite a OpDef to satisfy the SSA (Static Single Assignment)."""
        blob_names = context.blob_names
        blob_versions = context.blob_versions
        inputs, outputs = [], []
        for name in op_def.input:
            inputs.append(blob_names[name] if name in blob_names else name)
        for name in op_def.output:
            outputs.append(name + '_%d' % blob_versions[name]
                           if blob_versions[name] > 0 else name)
            if name != '':
                blob_versions[name] += 1
            blob_names[name] = outputs[-1]
        op_def.ClearField('input')
        op_def.ClearField('output')
        op_def.input.extend(inputs)
        op_def.output.extend(outputs)

    @classmethod
    def _make_node(cls, op_def, context):
        """Return a NodeProto from the OpDef."""
        translate_fn = None
        getter = export_util._GLOBAL_REGISTERED_EXPORTERS.try_get
        # Select the last versioned exporter if necessary.
        for i in range(context.opset_version, 0, -1):
            versioned_op_type = op_def.type + '-%d' % i
            if getter(versioned_op_type) is not None:
                translate_fn = getter(versioned_op_type)
                break
        if translate_fn is None:
            if getter(op_def.type) is not None:
                # Use the non-versioned exporter.
                translate_fn = getter(op_def.type)
            else:
                # Fallback to the generic exporter.
                translate_fn = export_util.translate
        nodes, const_tensors = translate_fn(op_def, context)
        return nest.flatten(nodes), const_tensors


def record():
    """Context-manger to record the graph.

    Examples:

    ```python
    with dragon.onnx.record():
        ...
    ```

    See Also
    --------
    `dragon.onnx.export(...)`_

    """
    graph_tape = tapes.Tape()
    graph_tape._exporting = True
    return tapes._GLOBAL_TAPE_STACK.get_controller(graph_tape)


def export(
    inputs,
    outputs,
    f,
    input_names=None,
    output_names=None,
    input_shapes=None,
    opset_version=None,
    verbose=False,
    enable_onnx_checker=True,
):
    """Export the recorded graph to an onnx model.

    Enter into the record mode to export operators into an onnx model:

    ```python
    x = dragon.constant([1, 2, 3])
    with dragon.onnx.record():
        y = x * x
    dragon.onnx.export(inputs=[x], outputs=[y], f='model.onnx')
    ```

    Parameters
    ----------
    inputs : Union[Sequence, Dict]
        The model inputs.
    outputs : Union[Sequence, Dict]
        The model outputs.
    f : str
        The filename for exporting model.
    input_names : Sequence[str], optional
        The name to the inputs.
    output_names : Sequence[str], optional
        The name to the outputs.
    input_shapes : Union[Sequence, Dict], optional
        The optional rewritten for input shapes.
    opset_version : int, optional
        The version of operator set.
    verbose : bool, optional, default=False
        Whether to print the debug string of graph.
    enable_onnx_checker : bool, optional, default=True
        Whether to check if model is valid.

    """
    # Process the inputs.
    if isinstance(inputs, dict):
        if input_names is not None:
            raise ValueError(
                'Excepted the input names from <inputs>.\n'
                'You should set the <input_names> to None.')
        inputs, input_names = list(inputs.values()), list(inputs.keys())
    else:
        inputs = nest.flatten(inputs)

    # Process the outputs.
    if isinstance(outputs, dict):
        if output_names is not None:
            raise ValueError(
                'Excepted the output names from <outputs>.\n'
                'You should set the <output_names> to None.')
        outputs, output_names = list(outputs.values()), list(outputs.keys())
    else:
        outputs = nest.flatten(outputs)

    if eager_context.executing_eagerly():
        op_defs = []
        graph_tape = tapes.get_tape()
        if not hasattr(graph_tape, '_exporting'):
            raise RuntimeError('Please enter with ``onnx.frontend.record()``.')
        for op_def in graph_tape.get_elements():
            op_defs.append(dragon_pb2.OperatorDef())
            op_defs[-1].ParseFromString(op_def.SerializeAs())
        graph_def = dragon_pb2.GraphDef(op=op_defs)
    else:
        output_symbols = []
        for output in outputs:
            if types.is_tensor(output) and not output._is_variable:
                output_symbols.append(output)
        graph = GraphLib.from_outputs(output_symbols)
        graph.run()
        graph_def = graph._def
        graph_def.name = ''

    # Add inputs and outputs.
    for i, input in enumerate(inputs):
        if hasattr(input, 'id'):
            graph_def.input.extend([input.id])
        elif input_names is not None:
            graph_def.input.extend([input_names[i]])

    for i, output in enumerate(outputs):
        if hasattr(output, 'id'):
            graph_def.output.extend([output.id])
        elif output_names is not None:
            graph_def.output.extend([output_names[i]])

    # Make value info from inputs and outputs.
    value_names = graph_def.input[:] + graph_def.output[:]
    value_info = dict([(k, (helper.tensor_type(v.dtype), v.shape))
                       for k, v in zip(value_names, inputs + outputs)])

    # Extract the constants from inputs and outputs.
    constants = collections.OrderedDict()
    for k, v in zip(value_names, inputs + outputs):
        if isinstance(v, numpy.ndarray):
            constants[k] = v

    # Export.
    model = graph_def_to_onnx_model(
        graph_def=graph_def,
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes,
        constants=constants,
        value_info=value_info,
        opset_version=opset_version,
        workspace=workspace_util.get_workspace(),
        verbose=verbose,
        enable_onnx_checker=enable_onnx_checker,
    )
    serialization.save_bytes(serialization.serialize_proto(model), f)


graph_def_to_onnx_graph = DragonFrontend.graph_def_to_onnx_graph
graph_def_to_onnx_model = DragonFrontend.graph_def_to_onnx_model
