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
"""String utility"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy


def add_indent(s_, num_spaces):
    """Add indent to each line of the string."""
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def array_to_string(array, separator=', ', prefix='', suffix=''):
    """Return the debug string of array."""
    return prefix + numpy.array2string(array, separator=separator) + suffix
