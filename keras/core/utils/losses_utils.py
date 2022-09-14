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
"""Losses utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Reduction(object):
    """The reduction for cost."""

    NONE = 'none'
    SUM = 'sum'
    MEAN = 'mean'
    VALID = 'valid'

    @classmethod
    def all(cls):
        return cls.NONE, cls.SUM, cls.MEAN, cls.VALID

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError('Invalid Reduction Key %s.' % key)
