# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Execute tensor operations. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.util import six
from dragon.vm.torch.autograd import grad_mode
from dragon.vm.torch.cpp import device as device_cls
from dragon.vm.torch.jit import tape
from dragon.vm.torch.tensor import Tensor


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

    for input in inputs:
        input_names.append(input.id)
        if input.requires_grad:
            requires_grad = True

    requires_grad = requires_grad and grad_mode.is_grad_enabled()

    # Allocate outputs.
    ws = workspace.get_workspace()
    output_scope = context.get_eager_scope(requires_grad)
    gc = ws.collectors  # Garbage collectors

    for i, spec in enumerate(outputs):
        if isinstance(spec, six.string_types):
            output_names.append(spec)
        else:
            if isinstance(spec, device_cls):
                impl = ws.create_tensor(gc.TENSOR.alloc(output_scope))
                outputs[i] = Tensor(device=spec, gc=gc.TENSOR, impl=impl)
            output_names.append(outputs[i].id)

    # Generate the OpDef.
    default_tape = tape.get_default_tape()
    op_def = op_def.DeriveTo(input_names, output_names)

    # Record this operation for future developments.
    if default_tape is not None:
        default_tape.add_def(op_def)
        requires_grad = requires_grad or default_tape.retain_graph

    if len(inputs) > 0 and no_grad is False:
        if requires_grad:
            instance_tape = tape.Tape()
            for input in inputs:
                instance_tape.merge_from(input._tape)
                if not input._requires_grad:
                    instance_tape.add_empty_grad(input.id + '_grad')
            op_def.name = gc.OP.alloc(op_def.type)
            instance_tape.add_operation(op_def)
            for output in outputs:
                output._tape = instance_tape
                output._requires_grad = True
        else:
            if default_tape is not None and default_tape.retain_ops:
                op_def.name = gc.OP.alloc(op_def.type)
            for output in outputs:
                output._requires_grad = False

    # Dispatch the computation.
    if pre_callback is not None:
        pre_callback(ws, op_def.name)
    ws.run_operator(op_def)

    # Return the outputs.
    return outputs if len(outputs) > 1 else outputs[0]


def run_backward(tensors, grad_tensors=None, retain_graph=False):
    """Compute the gradients."""
    # Collect the volatiles and tape from tensors
    default_tape = tape.Tape()
    for i, tensor in enumerate(tensors):
        if not tensor._requires_grad:
            raise RuntimeError('Element %d of tensors does not require grad.' % i)
        default_tape.merge_from(tensor._tape)

    # Collect the grad from tensors
    input_grads = []
    if grad_tensors is not None:
        if len(grad_tensors) != len(tensors):
            raise ValueError('Number of tensors and grad tensors should be same.')
        for i, grad_tensor in enumerate(grad_tensors):
            if not isinstance(grad_tensor, Tensor):
                raise TypeError(
                    'Element {} of grad tensors should be a tensor, got {}.'
                    .format(i, type(grad_tensor).__name__))
            if grad_tensor.shape != tensors[i].shape:
                raise ValueError(
                    'Size of element {} of grad tensors should be {}, got {}.'
                    .format(i, tensors[i].shape, grad_tensor.shape))
            input_grads.append(grad_tensor.id)

    # Prepare resources to optimize the backward pass.
    op_defs = [v for k, v in sorted(default_tape.operations.items())]

    # Dispatch the backward execution.
    current_ws = workspace.get_workspace()
    current_ws.run_backward(
        op_defs=op_defs,
        targets=[tensor.id for tensor in tensors],
        sources=default_tape.sources,
        input_grads=input_grads,
        empty_grads=default_tape.empty_grads,
    )

    # Free the retained resources
    if not retain_graph:
        gc = current_ws.collectors
        for op_def in op_defs:
            gc.OP.collect(op_def.name)
            for output in op_def.output:
                if output not in op_def.input:
                    gc.TENSOR.collect(output)
