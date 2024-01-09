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
"""Registry utility."""

import collections
import functools


class Registry(object):
    """The base registry.py class."""

    def __init__(self, name):
        self._name = name
        self._registry = collections.OrderedDict()

    @property
    def keys(self):
        """Return all the registered keys."""
        return list(self._registry.keys())

    def get(self, name):
        """Return the registered function by name."""
        if not self.has(name):
            raise KeyError("`%s` is not registered in <%s>." % (name, self._name))
        return self._registry[name]

    def has(self, name):
        """Return a bool indicating the name is registered or not."""
        return name in self._registry

    def register(self, name, func=None, **kwargs):
        """Register a function by name."""

        def decorated(inner_function):
            for key in name if isinstance(name, (tuple, list)) else [name]:
                self._registry[key] = functools.partial(inner_function, **kwargs)
            return inner_function

        if func is not None:
            return decorated(func)
        return decorated

    def try_get(self, name):
        """Try to return the registered function by name."""
        if self.has(name):
            return self.get(name)
        return None
