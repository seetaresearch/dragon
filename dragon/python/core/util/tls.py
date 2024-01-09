# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""TLS utility."""

import contextlib
import copy
import threading


class Constant(threading.local):
    def __init__(self, **attrs):
        super(Constant, self).__init__()
        self.__dict__.update(attrs)


class Stack(threading.local):
    def __init__(self, defaults=None):
        super(Stack, self).__init__()
        self._enforce_nesting = True
        self.defaults = [] if defaults is None else defaults
        self.stack = copy.deepcopy(self.defaults)

    def get_default(self):
        return self.stack[-1] if len(self.stack) >= 1 else None

    def reset(self):
        self.stack = copy.deepcopy(self.defaults)

    def is_cleared(self):
        return not self.stack

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        self.stack.pop()

    @property
    def enforce_nesting(self):
        return self._enforce_nesting

    @enforce_nesting.setter
    def enforce_nesting(self, value):
        self._enforce_nesting = value

    @contextlib.contextmanager
    def get_controller(self, default):
        """A context manager for manipulating a default stack."""
        self.stack.append(default)
        try:
            yield default
        finally:
            if self.stack:
                if self._enforce_nesting:
                    if self.stack[-1] is not default:
                        raise RuntimeError("Nesting violated by the push or pop.")
                    self.stack.pop()
                else:
                    self.stack.remove(default)
