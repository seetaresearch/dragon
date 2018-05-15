# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/deprecation.py>
#
# ------------------------------------------------------------

import functools
import re

_PRINT_DEPRECATION_WARNINGS = True


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


def deprecated(date, instructions):
   _validate_deprecation_args(date, instructions)

   def deprecated_wrapper(func):
       _validate_callable(func, 'deprecated')
       @functools.wraps(func)
       def new_func(*args, **kwargs):
           from dragon.config import logger
           if _PRINT_DEPRECATION_WARNINGS:
               logger.warning(
                   '{} (from {}) is deprecated and will be removed {}.\n'
                   'Instructions for updating:\n{}'.
                       format(_get_qualified_name(func),
                              func.__module__,
                              'in a future version' if date is None else ('after %s' % date),
                              instructions))
               return func(*args, **kwargs)
       return new_func
   return deprecated_wrapper