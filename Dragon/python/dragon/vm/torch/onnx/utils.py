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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import OrderedDict

from dragon.proto.dragon_pb2 import OperatorDef, GraphDef
from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.jit import enforce_jit

from dragon.vm.onnx.frontend import DragonFrontend
from dragon.vm.onnx import make_value_info
from dragon.vm.onnx import surgery_on_graph_def
from dragon.vm.onnx import export_from_graph_def


def export(
    model, args, f, verbose=False,
        input_names=None, output_names=None,
           opset_version=None,
):
    """Export a model into ONNX format.

    The outputs will be obtained by calling ``model(*args)``,
    both the Tensor or numpy array are allowed.

    You can either specify the ``input_names``, or pass a *dict*
    to the ``args``. In the same way, ``model`` could return a *dict*
    to specify the ``output_names``.

    Also note that if a numpy array is given or returned,
    it's name is definitely required. Otherwise, ONNX can't
    export this value due to the lacking of *id*.

    Parameters
    ----------
    model : dragon.vm.torch.nn.Module
        The module to export.
    args : sequence or dict
        The inputs.
    f : str
        The exporting file name.
    verbose : boolean, optional, default=False
        Whether to print the ONNX graph.
    input_names : sequence of str, optional
        The names to the input.
    output_names : sequence of str, optional
        The names to the output.
    opset_version : int, optional, default=9
        The opset version.

    Returns
    -------
    onnx_pb2.ModelProto
        The ONNX model.

    """
    # 1) Process the inputs
    if isinstance(args, dict):
        if input_names is not None:
            raise ValueError(
                'Excepted the input names from ``args``.\n'
                'You should set the ``input_names`` to None.')
        inputs, input_names, args = \
            list(args.values()), list(args.keys()), [args]
    else:
        if not isinstance(args, (tuple, list)):
            args = [args]
        inputs = args

    # 2) Run the model to get the outputs
    with enforce_jit(): outputs = model(*args)

    # 3) Process the outputs
    if isinstance(outputs, dict):
        if output_names is not None:
            raise ValueError(
                'Excepted the output names from ``outputs``.\n'
                'You should set the ``output_names`` to None.')
        outputs, output_names = list(outputs.values()), list(outputs.keys())
    else:
        if not isinstance(outputs, (tuple, list)): outputs = [outputs]

    # 4) Set the op set version explicitly
    if opset_version is not None:
        DragonFrontend.target_opset_version = opset_version

    # 5) Collect operators
    all_expr = {}
    for output in outputs:
        if isinstance(output, Tensor):
            jit_recorder = output.__jit_recorder__
            if jit_recorder is not None:
                all_expr.update(jit_recorder.ops)
    all_expr = sorted(all_expr.items(), key=lambda d: d[0])
    forward_ops = [v for k, v in all_expr]

    # 6) OperatorDef => GraphDef
    graph_def = GraphDef()
    graph_def.name = 'PyTorch.Graph'
    forward_def = [OperatorDef() for _ in range(len(forward_ops))]
    for i in range(len(forward_ops)):
        forward_def[i].ParseFromString(
            forward_ops[i].SerializeAs())
    graph_def.op.extend(forward_def)

    # 7) Do the Surgery
    rename_dict = {}

    if input_names is not None:
        for idx, e in enumerate(inputs):
            if hasattr(e, 'name'):
                rename_dict[e.name] = input_names[idx]
    else:
        input_names = [e.name for e in inputs]

    if output_names is not None:
        for idx, e in enumerate(outputs):
            if hasattr(e, 'name'):
                rename_dict[e.name] = output_names[idx]
    else:
        output_names = [e.name for e in outputs]

    graph_def = surgery_on_graph_def(
        graph_def=graph_def,
        renamed_tensors=rename_dict,
        external_outputs=output_names,
    )

    # 8) Make value info from inputs and outputs
    value_info = dict([
        (k, make_value_info(v.shape, v.dtype))
            for k, v in zip(input_names, inputs)])
    value_info.update(dict([
        (k, make_value_info(v.shape, v.dtype))
            for k, v in zip(output_names, outputs)]))

    # 9) Extract the constants from inputs and outputs
    constants = OrderedDict()
    for k, v in zip(input_names, inputs):
        if isinstance(v, np.ndarray): constants[k] = v
    for k, v in zip(output_names, outputs):
        if isinstance(v, np.ndarray): constants[k] = v

    # 10) Export
    export_from_graph_def(
        graph_def=graph_def,
        f=f,
        verbose=verbose,
        constants=constants,
        value_info=value_info,
        enforce_no_running=True,
    )