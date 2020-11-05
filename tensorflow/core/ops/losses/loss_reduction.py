# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/losses/loss_reduction.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Reduction(object):
    """The reduction for losses."""

    NONE = 'none'
    SUM = 'sum'
    MEAN = 'mean'
    VALID = 'valid'

    @classmethod
    def all(cls):
        """
        Returns a list of the class objects.

        Args:
            cls: (todo): write your description
        """
        return cls.NONE, cls.SUM, cls.MEAN, cls.VALID

    @classmethod
    def validate(cls, key):
        """
        Validate the given key.

        Args:
            cls: (callable): write your description
            key: (str): write your description
        """
        if key not in cls.all():
            raise ValueError('Invalid Reduction Key %s.' % key)
