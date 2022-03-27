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
"""PyTorch ONNX frontend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy

from dragon.core.framework import tapes
from dragon.core.framework import workspace
from dragon.core.proto import dragon_pb2
from dragon.core.util import nest
from dragon.core.util import serialization
from dragon.vm.onnx.core import helper
from dragon.vm.onnx.core.frontend.native import graph_def_to_onnx_model


def export(
    model,
    args,
    f,
    input_names=None,
    output_names=None,
    input_shapes=None,
    opset_version=None,
    verbose=False,
    enable_onnx_checker=True,
):
    """Export the recorded graph to an onnx model.

    The outputs will be obtained by calling ``model(*args)``,
    both the tensor or numpy array are allowed:

    ```python
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.fc = torch.nn.Linear(3, 3)

        def forward(self, x):
            y = self.fc(x)
            return y, np.ones((2, 3))

    m = MyModule()
    x = torch.zeros(2, 3)
    torch.onnx.export(
        m,
        args=(x,),
        f='my_module.onnx',
        input_names=('x',),
        output_names=('y', 'ones'),
    )
    ```

    You can either specify the ``input_names``, or pass a *dict*
    to the ``args``. In the same way, ``model`` could return a *dict*
    to specify the ``output_names``:

    ```python
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.fc = torch.nn.Linear(3, 3)

        def forward(self, inputs):
            y = self.fc(inputs['x'])
            return {'y': y, 'ones': np.ones((2, 3))}

    m = MyModule()
    x = torch.zeros(2, 3)
    torch.onnx.export(
        m,
        args={'x': x},
        f='my_module.onnx',
    )
    ```

    Also note that if a numpy array is given or returned,
    it's name is definitely required. Otherwise, ONNX can't
    export this value due to the lacking of *id*.

    Parameters
    ----------
    model : dragon.vm.torch.nn.Module
        The module to export.
    args : Union[Sequence, Dict]
        The model inputs.
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
    if isinstance(args, dict):
        if input_names is not None:
            raise ValueError(
                'Excepted the input names from <args>.\n'
                'You should set the <input_names> to None.')
        inputs, input_names, args = \
            list(args.values()), list(args.keys()), [args]
    else:
        inputs = args = nest.flatten(args)

    # Run the model to get the outputs.
    graph_tape = tapes.Tape()
    graph_tape._tracing = True  # Enable tracing.
    graph_tape._exporting = True  # Enable exporting.
    with graph_tape:
        outputs = model(*args)

    # Process the outputs
    if isinstance(outputs, dict):
        if output_names is not None:
            raise ValueError(
                'Excepted the output names from <outputs>.\n'
                'You should set the <output_names> to None.')
        outputs, output_names = list(outputs.values()), list(outputs.keys())
    else:
        outputs = nest.flatten(outputs)

    # Make graph def.
    ops_def, graph_def = [], dragon_pb2.GraphDef()

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

    # Add operators.
    for op_def in graph_tape.get_elements():
        ops_def.append(dragon_pb2.OperatorDef())
        ops_def[-1].ParseFromString(op_def.SerializeAs())
    graph_def.op.extend(ops_def)

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
        workspace=workspace.get_workspace(),
        verbose=verbose,
        enable_onnx_checker=enable_onnx_checker,
    )
    serialization.save_bytes(serialization.serialize_proto(model), f)
