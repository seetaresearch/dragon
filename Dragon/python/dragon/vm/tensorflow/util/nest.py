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
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/nest.py>
#
# ------------------------------------------------------------

import collections as _collections
import six as _six


def is_sequence(seq):
    if isinstance(seq, dict):
        return True
    return (isinstance(seq, _collections.Sequence)
            and not isinstance(seq, _six.string_types))


def _yield_value(iterable):
  if isinstance(iterable, dict):
    for key in sorted(_six.iterkeys(iterable)):
      yield iterable[key]
  else:
    for value in iterable:
      yield value


def _yield_flat_nest(nest):
  for n in _yield_value(nest):
    if is_sequence(n):
      for ni in _yield_flat_nest(n):
        yield ni
    else:
      yield n


def flatten(nest):
    if is_sequence(nest):
        return list(_yield_flat_nest(nest))
    else:
        return [nest]
