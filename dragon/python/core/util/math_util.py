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
"""Math utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator


def prod(input):
    """Apply the product reduce to the input."""
    if len(input) == 0:
        return 1
    return functools.reduce(operator.mul, input)


def div_up(a, b):
    """Return the upper bound of a divide operation."""
    return (a + b - 1) // b
