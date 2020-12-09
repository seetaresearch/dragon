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
"""Deprecation utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from dragon.core.util import decorator
from dragon.core.util import logging

# Allow deprecation warnings to be silenced temporarily with a context manager.
_PRINT_DEPRECATION_WARNINGS = True

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


def _validate_callable(func, decorator_name):
    if not hasattr(func, '__call__'):
        raise ValueError(
            '%s is not a function. If this is a property, make sure'
            ' @property appears before @%s in your source code:'
            '\n\n@property\n@%s\ndef method(...)' % (
                func, decorator_name, decorator_name))


def _validate_deprecation_args(date, instructions):
    if date is not None and not re.match(r'20\d\d-[01]\d-[0123]\d', date):
        raise ValueError('Date must be YYYY-MM-DD.')
    if not instructions:
        raise ValueError('Don\'t deprecate things without conversion instructions!')


def _get_qualified_name(function):
    # Python 3
    if hasattr(function, '__qualname__'):
        return function.__qualname__
    # Python 2
    if hasattr(function, 'im_class'):
        return function.im_class.__name__ + '.' + function.__name__
    return function.__name__


def deprecated(date, instructions, warn_once=True):
    _validate_deprecation_args(date, instructions)

    def decorated(inner_func):
        _validate_callable(inner_func, 'deprecated')

        def wrapper(*args, **kwargs):
            if _PRINT_DEPRECATION_WARNINGS:
                if inner_func not in _PRINTED_WARNING:
                    if warn_once:
                        _PRINTED_WARNING[inner_func] = True
                    logging.warning(
                        '{} (from {}) is deprecated and will be removed {}.\n'
                        'Instructions for updating:\n{}'.format(
                            _get_qualified_name(inner_func),
                            inner_func.__module__,
                            'in a future version' if date is None else ('after %s' % date),
                            instructions))
                return inner_func(*args, **kwargs)

        return decorator.make_decorator(inner_func, wrapper)

    return decorated


def not_installed(package=''):
    """Return a dummy function for the package that is not installed."""
    def dummy_fn(*args, **kwargs):
        raise ImportError('Package <%s> is required but not installed.' % package)
    return dummy_fn


class NotInstalled(object):
    """Return a dummy object for the package that is not installed."""

    def __init__(self, package=''):
        self._package = package

    def __getattr__(self, item):
        raise ImportError('Package <%s> is required but not installed.' % self._package)
