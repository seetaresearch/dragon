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
"""Checkpoint utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import context
from dragon.core.framework import proto_util
from dragon.core.framework import tapes
from dragon.core.framework import workspace
from dragon.core.util import decorator
from dragon.core.util import nest
from dragon.vm.torch.core.autograd import grad_mode
from dragon.vm.torch.core.tensor import Tensor
from dragon.vm.torch.core.nn.modules.container import Sequential


class CheckpointFunction(object):
    """Checkpointing function."""

    @staticmethod
    def apply(function, *args, **kwargs):
        """Apply function and create a checkpoint."""
        kwargs.pop('preserve_rng_state', True)
        variable_scope = kwargs.pop('variable_scope', 'Buffer')
        original_variable_scope = context.get_variable_scope(True)
        if kwargs:
            raise ValueError('Unexpected keyword arguments: ' +
                             ','.join(arg for arg in kwargs))

        # Run function.
        graph_tape = tapes.Tape()
        graph_tape._tracing = True  # Enable tracing.
        graph_tape._checkpointing = True  # Enable checkpointing.
        graph_tape._original_variable_scope = original_variable_scope
        with grad_mode.no_grad(), graph_tape:
            with context.variable_scope(variable_scope):
                outputs = function(*args)

        # Collect involving tensors.
        tensor_inputs, tensor_outputs = [], []
        for arg in args:
            if isinstance(arg, Tensor):
                tensor_inputs.append(arg)
        for arg in nest.flatten(outputs):
            if isinstance(arg, Tensor):
                tensor_outputs.append(arg)

        # Fill tape with function context.
        op_tape = tapes.OrderedTape()
        op_handle = workspace.get_workspace().create_handle('Checkpoint')
        op_tape.add_element(proto_util.make_operator_def(
            op_type='Checkpoint',
            name=op_handle,
            inputs=[input.id for input in tensor_inputs],
            outputs=[output.id for output in tensor_outputs],
            defs=[v.SerializeAs() for v in graph_tape.get_elements()],
            buffer_scope=variable_scope,
            to_impl=True))
        op_tape.add_handle(op_handle)
        op_tape.merge_handles(graph_tape.get_handles())

        # Save input tensors for backward.
        for input in tensor_inputs + graph_tape.get_sources():
            op_tape.add_source(input)

        # Save tape for backward.
        for output in tensor_outputs:
            output._tape = op_tape
            output._requires_grad = True

        return outputs


def checkpoint(function, *args, **kwargs):
    """Apply function and create a checkpoint.

    Parameters
    ----------
    function : callable
        The function to apply.

    Returns
    -------
    Any
        The function outputs.

    """
    if not grad_mode.is_grad_enabled():
        return function(*args, **kwargs)
    return CheckpointFunction.apply(function, *args, **kwargs)


def checkpoint_sequential(functions, input, segments=1, **kwargs):
    """Apply functions and create segmental checkpoints.

    Parameters
    ----------
    functions : Union[torch.nn.Sequential, Sequence[callable]]
        The functions to apply sequentially.
    input : dragon.vm.torch.Tensor
        The input tensor.
    segments : Union[int, Sequence[int]], optional
        The number or size of chunked checkpoints.

    Returns
    -------
    Any
        The function outputs.

    """
    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end):
                input = functions[j](input)
            with no_checkpoint():
                input = functions[end](input)
            return input
        return forward

    preserve_rng_state = kwargs.pop('preserve_rng_state', True)
    variable_scope = kwargs.pop('variable_scope', 'Buffer')
    if kwargs:
        raise ValueError('Unexpected keyword arguments: ' +
                         ','.join(arg for arg in kwargs))

    if isinstance(functions, Sequential):
        functions = list(functions.children())

    start, end = 0, len(functions) - 1
    if not grad_mode.is_grad_enabled():
        return run_function(start, end, functions)(input)

    if nest.is_sequence(segments):
        size_segments = segments
        if sum(size_segments) != len(functions):
            raise ValueError('Failed to chunk {} functions into {} segments.'
                             .format(len(functions), segments))
    else:
        size = (len(functions) + segments - 1) // segments
        last_size = len(functions) - size * (segments - 1)
        if last_size <= 0:
            raise ValueError('Failed to chunk {} functions into {} segments.'
                             .format(len(functions), segments))
        size_segments = [size] * (segments - 1) + [last_size]

    for size in size_segments:
        end = start + size - 1
        input = checkpoint(
            run_function(start, end, functions), input,
            preserve_rng_state=preserve_rng_state,
            variable_scope=variable_scope)
        start = end + 1
    return input


class no_checkpoint(decorator._DecoratorContextManager):
    """Context-manager to disable checkpointing."""

    def __init__(self):
        """Create a ``no_checkpoint`` context manager."""
        self._checkpointing = False

    def __enter__(self):
        graph_tape = tapes.get_tape()
        if hasattr(graph_tape, '_checkpointing'):
            self._checkpointing = True
            context._GLOBAL_VARIABLE_SCOPE_STACK.push(
                graph_tape._original_variable_scope)

    def __exit__(self, *args):
        if self._checkpointing:
            self._checkpointing = False
            context._GLOBAL_VARIABLE_SCOPE_STACK.pop()
        return False
