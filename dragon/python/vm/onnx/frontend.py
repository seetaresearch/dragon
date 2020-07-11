# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/pytorch/pytorch/blob/master/caffe2/python/onnx/frontend.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

try:
    import onnx
except ImportError:
    onnx = None

from dragon.core.util import nest
from dragon.vm.onnx import exporter
from dragon.vm.onnx import helper


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
        ws=None,
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
            ws.RegisterAlias(graph_def.output[i], alias)
            if graph_def.output[i] in value_info:
                value_info[alias] = value_info[graph_def.output[i]]
        for i, alias in enumerate(input_names):
            blob_aliases[graph_def.input[i]] = alias
            ws.RegisterAlias(graph_def.input[i], alias)
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
            name=graph_def.name
            if len(graph_def.name) > 0
            else 'onnx-model'
        )

        graph_inputs = helper.collect_inputs(graph_def)

        # Add nodes.
        shapes, blob_names, initializers = {}, {}, []
        blob_versions = collections.defaultdict(
            int, **dict((blob_aliases.get(k, k), 1) for k in graph_inputs))

        for op in graph_def.op:
            # Get the shape of inputs and outputs.
            for name in itertools.chain(op.input, op.output):
                impl = ws.GetTensor(name)
                if impl is not None:
                    shapes[name] = impl.dims
                else:
                    shapes[name] = value_info[name][1]

            # Translate definition.
            nodes, const_tensors = cls._translate(op, opset_version, shapes, ws)

            # Rewritten for names.
            for node in nodes:
                node.input[:] = [blob_aliases.get(e, e) for e in node.input]
                node.output[:] = [blob_aliases.get(e, e) for e in node.output]
                node, blob_names, blob_versions = \
                    cls._ssa_rewrite(node, blob_names, blob_versions)

            # Directly convert outputs as const tensors if necessary.
            if None in nodes:
                const_tensors = [helper.from_tensor(name, ws) for name in op.output]
            else:
                onnx_graph.node.extend(nodes)

            # Merge constant tensors.
            if const_tensors is not None:
                value_info = {
                    **value_info,
                    **dict((
                        e.name, (e.data_type, e.dims)
                    ) for e in const_tensors)
                }
                initializers.extend(const_tensors)

        # Add constants.
        if constants is not None:
            for k, v in constants.items():
                initializers.append(helper.from_array(v, name=k))

        # Add initializers.
        onnx_graph.initializer.extend(initializers)

        # Add inputs.
        for name in helper.collect_inputs(onnx_graph):
            try:
                onnx_graph.input.extend([
                    helper.make_tensor_value_info(
                        name=name,
                        elem_type=value_info[name][0],
                        shape=value_info[name][1],
                    )
                ])
            except KeyError:
                raise ValueError(
                    'Info of tensor `{}` is missing, '
                    'specify it in <value_info>.'.format(name)
                )

        # Add outputs.
        onnx_graph.output.extend(
            helper.make_tensor_value_info(
                name=blob_names.get(name, name),
                elem_type=value_info[name][0],
                shape=value_info[name][1],
            ) for name in [blob_aliases.get(e, e) for e in set(graph_def.output)]
        )

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
        if onnx.__version__ < onnx_version:
            raise RuntimeError(
                'OpSet {} requires ONNX version >= {}. '
                '({} currently installed.)'
                .format(opset_version, onnx_version, onnx.__version__)
            )
        return opset_version

    @staticmethod
    def _ssa_rewrite(op_def, blob_names, blob_versions):
        """Rewrite a OpDef to satisfy the SSA (Static Single Assignment)."""
        inputs, outputs = [], []
        for e in op_def.input:
            inputs.append(blob_names[e] if e in blob_names else e)
        for e in op_def.output:
            outputs.append(e + '_%d' % blob_versions[e]
                           if blob_versions[e] > 0 else e)
            blob_versions[e] += 1
            blob_names[e] = outputs[-1]
        op_def.ClearField('input')
        op_def.ClearField('output')
        op_def.input.extend(inputs)
        op_def.output.extend(outputs)
        return op_def, blob_names, blob_versions

    @classmethod
    def _translate(cls, op_def, opset_version, shape_dict, ws):
        """Return a NodeProto from the OpDef."""
        translate_fn = None
        getter = exporter._GLOBAL_REGISTERED_EXPORTERS.try_get
        # Select the last versioned exporter if necessary.
        for i in range(opset_version, 0, -1):
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
                translate_fn = exporter.translate
        nodes, const_tensors = translate_fn(op_def, shape_dict, ws)
        return nest.flatten(nodes), const_tensors


graph_def_to_onnx_graph = DragonFrontend.graph_def_to_onnx_graph
graph_def_to_onnx_model = DragonFrontend.graph_def_to_onnx_model
