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

"""Recording tape utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import tls


def _new_incrementer():
    """Return a incrementer from 1."""
    i = 0  # Python returns BigInteger.
    while True:
        i += 1
        yield i


class Tape(object):
    """Record the executed operations."""

    UID_GENERATOR = _new_incrementer()

    def __init__(self, retain_ops=False, retain_graph=False):
        self._defs = []
        self._operations = dict()
        self.retain_ops = retain_ops
        self.retain_graph = retain_graph

    @property
    def defs(self):
        """Return the recording defs."""
        return self._defs

    @property
    def operations(self):
        """Return the recording operations."""
        return self._operations

    def add_def(self, op_def):
        """Add a new def."""
        self._defs.append(op_def)

    def add_operation(self, op_def):
        """Add a new operation."""
        uid = next(self.UID_GENERATOR)
        self._operations[uid] = op_def

    def merge_from(self, other):
        """Merge operations from another."""
        if other is not None:
            self._operations = {**self._operations, **other._operations}

    def __enter__(self):
        """Enter the tape into the stack."""
        _GLOBAL_TAPE_STACK.push(self)
        return self

    def __exit__(self, typ, value, traceback):
        """Exit the tape from the stack."""
        _GLOBAL_TAPE_STACK.pop()


def get_default_tape():
    """Return the current default tape."""
    return _GLOBAL_TAPE_STACK.get_default()


# Define a global stack to store the tapes of current thread
_GLOBAL_TAPE_STACK = tls.Stack()
