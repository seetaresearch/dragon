# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/backprop.py>
#
# ------------------------------------------------------------
"""Back-propagation engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from dragon.core.autograph import tape
from dragon.core.autograph import context
from dragon.core.framework import device_spec
from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor
from dragon.core.util import nest


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
        """Compute the derivatives of target w.r.t. sources."""
        # Check the pushed tape.
        if self._tape is None:
            raise RuntimeError(
                'GradientTape.gradient(...) can only be called '
                'once on non-persistent tapes.')

        # Stop recording if not persistent.
        if self._recording:
            if not self._persistent:
                self._pop_tape()

        # Collect gradient info.
        xs, ys, grad_ys = nest.flatten(sources), nest.flatten(target), []
        if output_gradients is not None:
            for tensor, grad_tensor in zip(ys, nest.flatten(output_gradients)):
                if grad_tensor.shape != tensor.shape:
                    raise ValueError(
                        'Excepted the dimensions of output gradient is {}, '
                        'got {}.'.format(tensor.shape, grad_tensor.shape))
                grad_ys.append(grad_tensor)

        # Record or execute the computation.
        execute_ws = workspace.get_workspace()
        if not context.executing_eagerly():
            grad_xs = []
            for x in xs:
                self._tape.add_source(x)
                grad_xs.append(Tensor(
                    shape=x.shape,
                    dtype=x.dtype,
                    impl=execute_ws.create_tensor(x.id + '_grad'),
                    symbolic=True))
            for i, y in enumerate(ys):
                y._grad_tape = self._tape
                y._grad = grad_ys[i] if i < len(grad_ys) else None
            if not self._persistent:
                self._tape = None
            return grad_xs
        else:
            execute_ws.run_backward(
                targets=[y.id for y in ys],
                sources=[x.id for x in xs],
                grad_targets=[dy.id for dy in grad_ys],
                op_defs=self._tape.get_op_defs())
            # Remove the tape.
            if not self._persistent:
                self._tape.release(execute_ws)
                self._tape = None
            # Pack the gradients.
            return [self._get_grad(execute_ws, x) for x in xs]

    def reset(self):
        """Destroy the tape and push a new one."""
        self._pop_tape()
        self._tape.release()
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
            The tensor to be traced.

        """
        # Check the pushed tape.
        if self._tape is None:
            raise RuntimeError(
                'GradientTape.gradient can only be called '
                'once on non-persistent tapes.')
        for var in nest.flatten(tensor):
            self._tape.add_target(id(var))

    def _pop_tape(self):
        """Pop the pushed tape."""
        if not self._recording:
            raise ValueError('Tape is not recording.')
        tape.pop_tape()
        self._recording = False

    def _push_tape(self):
        """Push the tape."""
        if self._recording:
            raise ValueError('Tape is already recording.')
        if self._tape is None:
            self._tape = tape.Tape()
        tape.push_new_tape(self._tape)
        self._recording = True

    @staticmethod
    def _get_grad(execute_ws, source):
        """Return the grad tensor of given source."""
        impl = execute_ws.get_tensor(source.id + '_grad')
        if impl is None:
            return None
        device = device_spec.DeviceSpec(*impl.device)
        return Tensor(shape=None, impl=impl, device=device)

    def __del__(self):
        if self._tape is not None:
            self._tape.release()

    def __enter__(self):
        """Enter the tape into the stack."""
        self._push_tape()
        return self

    def __exit__(self, typ, value, traceback):
        """Exit the tape from the stack."""
        if self._recording:
            self._pop_tape()
