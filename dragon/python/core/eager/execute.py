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
"""Execute tensor operations. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.eager import backprop
from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import device_spec
from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.util import six


def run_operator(
    op_def,
    inputs,
    outputs,
    no_grad=False,
    pre_callback=None,
):
    """Compute the outputs."""
    requires_grad = False
    input_names, output_names = [], []
    default_tape = backprop.get_default_tape()

    # Add inputs.
    for input in inputs:
        input_names.append(input.id)
        if default_tape is not None:
            if input.requires_grad:
                requires_grad = True
            elif default_tape.is_watched(input):
                requires_grad = True
            else:
                default_tape.add_empty_grad(input.id + '_grad')

    # Determine the gradient flags.
    if default_tape is not None:
        no_grad = no_grad and not default_tape.retain_graph
        requires_grad = requires_grad or default_tape.retain_graph

    # Allocate outputs.
    ws = workspace.get_workspace()
    output_scope = context.get_eager_scope(requires_grad)
    gc = ws.collectors  # Garbage collectors

    for i, spec in enumerate(outputs):
        if isinstance(spec, six.string_types):
            output_names.append(spec)
        else:
            if isinstance(spec, device_spec.DeviceSpec):
                impl = ws.create_tensor(gc.TENSOR.alloc(output_scope))
                outputs[i] = EagerTensor(device=spec, gc=gc.TENSOR, impl=impl)
            output_names.append(outputs[i].id)

    # Generate OpDef.
    op_handle = None  # Optional resource handle
    op_def = op_def.DeriveTo(input_names, output_names)

    # Flag the outputs.
    if len(inputs) > 0 and not no_grad:
        if requires_grad:
            for output in outputs:
                output._requires_grad = True
            op_def.name = op_handle = gc.OP.alloc(op_def.type)
            default_tape.add_def(op_def)
        else:
            for output in outputs:
                output._requires_grad = False

    # Retain the handle if necessary.
    if default_tape is not None:
        if default_tape.retain_op_handles and op_handle is None:
            op_def.name = gc.OP.alloc(op_def.type)

    # Dispatch the computation.
    if pre_callback is not None:
        pre_callback(ws, op_def.name)
    ws.run_operator(op_def)

    # Return the outputs.
    return outputs[0] if len(outputs) == 1 else outputs
