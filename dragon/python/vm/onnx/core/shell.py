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
"""Toolkit for manipulating the onnx api."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy

from dragon.core.autograph import function_lib
from dragon.core.eager import context
from dragon.core.eager import backprop
from dragon.core.framework import types
from dragon.core.framework import workspace
from dragon.core.proto import dragon_pb2
from dragon.core.util import nest
from dragon.vm.onnx.core import utils as onnx_util


class Shell(object):
    """Context-manger to export or load onnx models.

    Enter a shell to export operators into an onnx model:

    ```python
    x = dragon.constant([1, 2, 3])
    with onnx.Shell() as shell, shell.as_default():
        y = x * x
    shell.export(inputs=[x], outputs=[y], f='model.onnx')
    ```

    The onnx models can also be loaded to execute:

    ```python
    f = shell.load_model('model.onnx', explicit_inputs=True)
    print(f(np.array([1, 2, 3]))
    ```

    """

    def __init__(self):
        """Create a ``Shell``."""
        self._workspace = None
        self._tape = backprop.GradientTape()

    def as_default(self):
        """Set as the default shell."""
        return self._workspace.as_default()

    def export(
        self,
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
        """Export an onnx model.

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

        if context.executing_eagerly():
            # Make graph def.
            op_defs = []
            for op_def in self._tape._tape._defs:
                op_defs.append(dragon_pb2.OperatorDef())
                op_defs[-1].ParseFromString(op_def.SerializeAs())
            graph_def = dragon_pb2.GraphDef(op=op_defs)
        else:
            symbolic_outputs = []
            for output in outputs:
                if types.is_symbolic_tensor(output):
                    symbolic_outputs.append(output)
            with self.as_default():
                graph_func = function_lib.create_function(
                    outputs=symbolic_outputs)
                graph_func.callback()
                graph_def = graph_func.graph_def
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
        value_info = dict([
            (k, onnx_util.make_value_info(v.shape, v.dtype))
            for k, v in zip(value_names, inputs + outputs)
        ])

        # Extract the constants from inputs and outputs.
        constants = collections.OrderedDict()
        for k, v in zip(value_names, inputs + outputs):
            if isinstance(v, numpy.ndarray):
                constants[k] = v

        # Export.
        onnx_util.export_from_graph(
            graph_def=graph_def,
            f=f,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes,
            constants=constants,
            value_info=value_info,
            opset_version=opset_version,
            workspace=self._workspace,
            verbose=verbose,
            enable_onnx_checker=enable_onnx_checker,
        )

    @staticmethod
    def load_model(model_path, explicit_inputs=False):
        """Import an onnx model to the function.

        Parameters
        ----------
        model_path : str
            The path to the onnx model.
        explicit_inputs : bool, optional, default=False
            **True** to attach model inputs to the function.

        Returns
        -------
        callable
            The function to run the model once.

        """
        return onnx_util.import_to_function(model_path, explicit_inputs)

    def __enter__(self):
        self._workspace = workspace.Workspace()
        self._workspace.merge_from(workspace.get_workspace())
        if context.executing_eagerly():
            self._tape._push_tape()
            self._tape._tape.retain_graph = True
        return self

    def __exit__(self, typ, value, traceback):
        if self._tape._recording:
            self._tape._pop_tape()
