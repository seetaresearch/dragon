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
"""Decorator utility."""

import functools
import inspect
import sys


class _Decorator(object):
    """The metaclass of decorator objects."""

    def __init__(self, target):
        self._decorated_target = target


class _DecoratorContextManager(object):
    """The metaclass of decorator context manager."""

    def __call__(self, func):
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)

        return decorate_context

    def _wrap_generator(self, func):
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)
            cls = type(self)
            try:
                with cls():
                    response = gen.send(None)
                while True:
                    try:
                        request = yield response
                    except GeneratorExit:
                        with cls():
                            gen.close()
                        raise
                    except BaseException:
                        with cls():
                            response = gen.throw(*sys.exc_info())
                    else:
                        with cls():
                            response = gen.send(request)
            except StopIteration as e:
                return e.value

        return generator_context

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *args):
        raise NotImplementedError


def make_decorator(target, decorator_func):
    decorator = _Decorator(target)
    setattr(decorator_func, "_dragon_decorator", decorator)
    if hasattr(target, "__name__"):
        decorator_func.__name__ = target.__name__
    if hasattr(target, "__module__"):
        decorator_func.__module__ = target.__module__
    if hasattr(target, "__dict__"):
        for name in target.__dict__:
            if name not in decorator_func.__dict__:
                decorator_func.__dict__[name] = target.__dict__[name]
    if hasattr(target, "__doc__"):
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
        elif hasattr(cur, "_dragon_decorator") and isinstance(
            getattr(cur, "_dragon_decorator"), _Decorator
        ):
            decorators.append(getattr(cur, "_dragon_decorator"))
        else:
            break
        if not hasattr(decorators[-1], "_decorated_target"):
            break
        cur = decorators[-1]._decorated_target
    return decorators, cur
