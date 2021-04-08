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
"""Container to record the operators."""

from dragon.core.framework import workspace
from dragon.core.util import tls


def _new_incrementer():
    """Return a incrementer from 1."""
    i = 0  # Python returns big integer.
    while True:
        i += 1
        yield i


class Tape(object):
    """Record operators in a sequential container."""

    def __init__(self):
        self._op_defs = []
        self._sources = set()
        self._targets = set()

    def add_op_def(self, value):
        """Add a operator def."""
        self._op_defs.append(value)

    def add_source(self, value):
        """Add a source."""
        self._sources.add(value)

    def add_target(self, value):
        """Add a target."""
        self._targets.add(value)

    def get_op_defs(self):
        """Return the recorded operator defs."""
        return self._op_defs

    def get_sources(self):
        """Return the sources."""
        return list(self._sources)

    def get_targets(self):
        """Return the targets."""
        return list(self._targets)

    def is_source(self, value):
        """Return if value is a source."""
        return value in self._sources

    def is_target(self, value):
        """Return if value is a target."""
        return value in self._targets

    def merge_op_defs(self, op_defs):
        """Merge the operator defs."""
        self._op_defs.extend(op_defs)

    def merge_from(self, other):
        """Merge from the given tape."""
        if other is not None:
            self.merge_op_defs(other._op_defs)
            self._sources |= other._sources
            self._targets |= other._targets

    def release(self, execute_ws=None, op_defs=None):
        """Release the resources."""
        execute_ws = execute_ws or workspace.get_workspace()
        op_defs = op_defs or self.get_op_defs()
        handle_pool = execute_ws._handle_pool
        for op_def in op_defs:
            handle_pool.release(op_def.name)

    def __enter__(self):
        """Enter the tape into the stack."""
        _GLOBAL_TAPE_STACK.push(self)
        return self

    def __exit__(self, typ, value, traceback):
        """Exit the tape from the stack."""
        _GLOBAL_TAPE_STACK.pop()


class OrderedTape(Tape):
    """Record operators in an ordered container."""

    _op_index = _new_incrementer()

    def __init__(self):
        super(OrderedTape, self).__init__()
        self._op_defs = dict()

    def add_op_def(self, value):
        """Add a new operator def."""
        self._op_defs[next(self._op_index)] = value

    def get_op_defs(self):
        """Return the recorded operator defs."""
        return [v for k, v in sorted(self._op_defs.items())]

    def merge_op_defs(self, op_defs):
        """Merge the operator defs."""
        self._op_defs = {**self._op_defs, **op_defs}


class GraphTape(Tape):
    """Record operators in a sequential graph."""


def get_tape():
    """Return the current active tape."""
    return _GLOBAL_TAPE_STACK.get_default()


def push_new_tape(tape):
    """Push the tape into the global stack."""
    _GLOBAL_TAPE_STACK.push(tape)


def pop_tape():
    """Pop the tape in top of the stack."""
    _GLOBAL_TAPE_STACK.pop()


# Define a global stack to store the tapes of current thread.
_GLOBAL_TAPE_STACK = tls.Stack()
