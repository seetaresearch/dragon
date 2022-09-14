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
"""Container to record the elements."""

from dragon.core.util import tls


def _new_incrementer():
    """Return an incrementer from 1."""
    i = 0  # Python returns big integer.
    while True:
        i += 1
        yield i


class Tape(object):
    """Record elements in a sequential container."""

    def __init__(self):
        self._elements = []
        self._handles = set()
        self._sources = set()
        self._targets = set()

    def add_element(self, element):
        """Add an element."""
        self._elements.append(element)

    def add_handle(self, value):
        """Add a handle."""
        self._handles.add(value)

    def add_source(self, value):
        """Add a source."""
        self._sources.add(value)

    def add_target(self, value):
        """Add a target."""
        self._targets.add(value)

    def get_elements(self):
        """Return the elements."""
        return self._elements

    def get_handles(self):
        """Return the handles."""
        return list(self._handles)

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

    def merge_elements(self, elements):
        """Merge the given elements."""
        self._elements.extend(elements)

    def merge_handles(self, handles):
        """Merge the given handles."""
        self._handles |= set(handles)

    def merge_from(self, other):
        """Merge from the given tape."""
        if other is not None:
            self.merge_elements(other._elements)
            self.merge_handles(other._handles)
            self._sources |= other._sources
            self._targets |= other._targets

    def __enter__(self):
        """Enter the tape into the stack."""
        _GLOBAL_TAPE_STACK.push(self)
        return self

    def __exit__(self, typ, value, traceback):
        """Exit the tape from the stack."""
        _GLOBAL_TAPE_STACK.pop()


class OrderedTape(Tape):
    """Record elements in an ordered container."""

    _element_index = _new_incrementer()

    def __init__(self):
        super(OrderedTape, self).__init__()
        self._elements = dict()

    def add_element(self, element):
        """Add an element."""
        self._elements[next(self._element_index)] = element

    def get_elements(self):
        """Return the elements."""
        return [v for k, v in sorted(self._elements.items())]

    def merge_elements(self, elements):
        """Merge the given elements."""
        self._elements = {**self._elements, **elements}


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
