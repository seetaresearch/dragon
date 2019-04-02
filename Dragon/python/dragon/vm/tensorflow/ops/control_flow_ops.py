# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import ops as _ops


def equal(a, b, name=None):
    return _ops.Equal([a, b], name=name)


def greater(a, b, name=None):
    return _ops.Greater([a, b], name=name)


def less(a, b, name=None):
    return _ops.Less([a, b], name=name)

