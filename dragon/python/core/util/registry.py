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
"""Registry utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
            raise KeyError(
                "`%s` is not registered in <%s>."
                % (name, self._name))
        return self._registry[name]

    def has(self, name):
        """Return a bool indicating the name is registered or not."""
        return name in self._registry

    def register(self, name, func=None, **kwargs):
        """Register a function by name."""
        def decorated(inner_function):
            for key in (name if isinstance(
                    name, (tuple, list)) else [name]):
                self._registry[key] = functools.partial(
                    inner_function, **kwargs)
            return inner_function
        if func is not None:
            return decorated(func)
        return decorated

    def try_get(self, name):
        """Try to return the registered function by name."""
        if self.has(name):
            return self.get(name)
        return None
