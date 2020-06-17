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
#       <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/backprop.py>
#
# ------------------------------------------------------------

"""Do back-propagation from the eager expressions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from dragon.core.autograph import grad_impl
from dragon.core.eager import context
from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import device_spec
from dragon.core.framework import workspace
from dragon.core.util import nest
from dragon.core.util import tls


class Tape(object):
    def __init__(self, parent):
        self._defs = []
        self._parent = parent
        self._watched = set()
        self._gc = workspace.get_workspace().collectors
        self._retain_graph = False

    def add_def(self, op_def):
        """Add a new def."""
        self._defs.append(op_def)

    def is_watched(self, tensor):
        """Return true if tensor is watched."""
        return tensor.id in self._watched

    def watch(self, tensor):
        """Ensure the tensor will be traced."""
        self._watched.add(tensor.id)

    def __del__(self):
        """Release the resources."""
        for op_def in self._defs:
            self._gc.OPERATOR.collect(op_def.name)
            for y in op_def.output:
                if y not in op_def.input:
                    self._gc.TENSOR.collect(y)


class GradientTape(object):
    """Record the operations for auto differentiation.

    You should enter a tape before the execution performed:

    ```python
    with dragon.eager_mode():
        x = dragon.ones(shape=(2, 3))
        with dragon.GradientTape() as tape:
            y = x + 1
        print(tape.gradient(y, x))  # None, as ``x`` is not watched

        with dragon.GradientTape() as tape:
            tape.watch(x)
            y = x + 1
        print(tape.gradient(y, x))  # Ok
    ```

    """

    def __init__(self, persistent=False):
        """Create a ``GradientTape``.

        Parameters
        ----------
        persistent : bool, optional, default=False
            Whether to retain graph once ``gradient(...)`` called.

        """
        self._tape = None
        self._persistent = persistent
        self._recording = False

    def gradient(self, target, sources, output_gradients=None):
        """Compute the derivatives of ``target`` w.r.t. ``sources``."""
        # Fallback to the symbolic implementation.
        if not context.executing_eagerly():
            return grad_impl.gradients(
                ys=target,
                xs=sources,
                grad_ys=output_gradients,
            )

        # Check the pushed tape.
        if self._tape is None:
            raise RuntimeError(
                'GradientTape.gradient can only be called '
                'once on non-persistent tapes.'
            )
        if self._recording:
            if not self._persistent:
                self._pop_tape()

        # Collect gradient info.
        inputs, outputs = [], []
        targets, ignores = [], []
        target = nest.flatten(target)
        sources = nest.flatten(sources)
        sources_is_watched = []
        if output_gradients is not None:
            output_gradients = nest.flatten(output_gradients)
            for value, grad in zip(target, output_gradients):
                if grad.shape != value.shape:
                    raise ValueError(
                        'Except the dimensions of <output_gradient> is {}, '
                        'got {}.'.format(value.shape, grad.shape)
                    )
                inputs.append(grad.id)
        for t in target:
            targets.append(t.id)
        for s in sources:
            sources_is_watched.append(self._tape.is_watched(s))
            if not s.requires_grad and not sources_is_watched[-1]:
                ignores.append(s.id + '_grad')
            else:
                outputs.append(s.id)

        # Run the gradient ops sequentially.
        workspace.run_backward(
            forward_ops=self._tape._defs,
            targets=targets,
            sources=outputs,
            input_grads=inputs,
            ignored_grads=ignores,
        )

        # Remove the tape to release resources.
        if not self._persistent:
            self._tape = None

        # Pack the gradients.
        return [_steal_grad_ref(s, w) for s, w
                in zip(sources, sources_is_watched)]

    def replay(self):
        """Run the operators stored in the tape."""
        workspace.run_operator(self._tape._defs)

    def reset(self):
        """Destroy the tape and push a new one."""
        self._pop_tape()
        self._tape = None
        self._push_tape()

    @contextlib.contextmanager
    def stop_recording(self):
        """Temporarily stop the recording."""
        if self._tape is None:
            raise ValueError('Missing the recording tape.')
        self._pop_tape()
        try:
            yield
        finally:
            self._push_tape()

    def watch(self, tensor):
        """Ensure the input tensor will be traced.

        Parameters
        ----------
        tensor : Sequence[dragon.Tensor]
            The tensor(s) to be traced.

        """
        # Check the pushed tape.
        if self._tape is None:
            raise RuntimeError(
                'GradientTape.gradient can only be called '
                'once on non-persistent tapes.'
            )
        for t in nest.flatten(tensor):
            self._tape.watch(t)

    def _pop_tape(self):
        if not self._recording:
            raise ValueError('Tape is not recording.')
        pop_tape()
        self._recording = False

    def _push_tape(self):
        if self._recording:
            raise ValueError('Tape is already recording.')
        if self._tape is None:
            self._tape = Tape(self)
        push_new_tape(self._tape)
        self._recording = True

    def __enter__(self):
        """Enter the tape into the stack."""
        self._push_tape()
        return self

    def __exit__(self, typ, value, traceback):
        """Exit the tape from the stack."""
        if self._recording:
            self._pop_tape()


def get_default_tape():
    """Return the current active tape."""
    return _GLOBAL_TAPE_STACK.get_default()


def push_new_tape(tape):
    """Push the tape into the global stack."""
    _GLOBAL_TAPE_STACK.push(tape)


def pop_tape():
    """Pop the tape in top of the stack."""
    _GLOBAL_TAPE_STACK.pop()


def _steal_grad_ref(source, is_watched=False):
    if not source.requires_grad and not is_watched:
        return None
    grad_id = source.id + '_grad'
    grad_impl = workspace.get_workspace().GetTensor(grad_id)
    if grad_impl is None:
        return None
    device = device_spec.DeviceSpec(*grad_impl.device)
    grad_ref = EagerTensor(own_storage=False, device=device)
    grad_ref._id, grad_ref._impl = grad_id, grad_impl
    return grad_ref


# Define a global stack to store the tapes of current thread.
_GLOBAL_TAPE_STACK = tls.Stack()
