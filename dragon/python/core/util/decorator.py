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
"""Decorator utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class _Decorator(object):
    """The metaclass of decorator objects."""
    def __init__(self, target):
        self._decorated_target = target


def make_decorator(target, decorator_func):
    decorator = _Decorator(target)
    setattr(decorator_func, '_dragon_decorator', decorator)
    if hasattr(target, '__name__'):
        decorator_func.__name__ = target.__name__
    if hasattr(target, '__module__'):
        decorator_func.__module__ = target.__module__
    if hasattr(target, '__dict__'):
        for name in target.__dict__:
            if name not in decorator_func.__dict__:
                decorator_func.__dict__[name] = target.__dict__[name]
    if hasattr(target, '__doc__'):
        decorator_func.__doc__ = target.__doc__
    decorator_func.__wrapped__ = target
    decorator_func.__original_wrapped__ = target
    return decorator_func


def unwrap(maybe_decorator):
    """Unwrap the decorator recursively."""
    decorators = []
    cur = maybe_decorator
    while True:
        if isinstance(cur, _Decorator):
            decorators.append(cur)
        elif (hasattr(cur, '_dragon_decorator') and
                isinstance(getattr(cur, '_dragon_decorator'), _Decorator)):
            decorators.append(getattr(cur, '_dragon_decorator'))
        else:
            break
        if not hasattr(decorators[-1], '_decorated_target'):
            break
        cur = decorators[-1]._decorated_target
    return decorators, cur
