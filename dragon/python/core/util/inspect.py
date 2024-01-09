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
"""Inspect utility"""

import collections
import inspect as _inspect

from dragon.core.util import decorator


ArgSpec = _inspect.ArgSpec


if hasattr(_inspect, "FullArgSpec"):
    FullArgSpec = _inspect.FullArgSpec
else:
    FullArgSpec = collections.namedtuple(
        "FullArgSpec",
        [
            "args",
            "varargs",
            "varkw",
            "defaults",
            "kwonlyargs",
            "kwonlydefaults",
            "annotations",
        ],
    )


def _convert_maybe_argspec_to_fullargspec(argspec):
    if isinstance(argspec, FullArgSpec):
        return argspec
    return FullArgSpec(
        args=argspec.args,
        varargs=argspec.varargs,
        varkw=argspec.keywords,
        defaults=argspec.defaults,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={},
    )


if hasattr(_inspect, "getfullargspec"):
    _getfullargspec = _inspect.getfullargspec

    def _getargspec(target):
        """A python3 version of getargspec."""
        fullargspecs = getfullargspec(target)
        return ArgSpec(
            args=fullargspecs.args,
            varargs=fullargspecs.varargs,
            keywords=fullargspecs.varkw,
            defaults=fullargspecs.defaults,
        )

else:
    _getargspec = _inspect.getargspec

    def _getfullargspec(target):
        """A python2 version of getfullargspec."""
        return _convert_maybe_argspec_to_fullargspec(getargspec(target))


def getargspec(obj):
    """Decorator-aware replacement for ``inspect.getargspec``."""
    _, target = decorator.unwrap(obj)
    return _getargspec(target)


def getfullargspec(obj):
    """Decorator-aware replacement for ``inspect.getfullargspec``."""
    _, target = decorator.unwrap(obj)
    return _getfullargspec(target)


def isclass(object):
    """Decorator-aware replacement for ``inspect.isclass``."""
    return _inspect.isclass(decorator.unwrap(object)[1])


def isfunction(object):
    """Decorator-aware replacement for ``inspect.isfunction``."""
    return _inspect.isfunction(decorator.unwrap(object)[1])


def ismethod(object):
    """Decorator-aware replacement for ``inspect.ismethod``."""
    return _inspect.ismethod(decorator.unwrap(object)[1])
