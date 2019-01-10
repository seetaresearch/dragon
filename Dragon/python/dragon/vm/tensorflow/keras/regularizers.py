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
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/regularizers.py>
#
# ------------------------------------------------------------

"""Built-in regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Regularizer(object):
    """Regularizer base class."""

    def __call__(self, x):
        return 0.


class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.

    Parameters
    ----------
    l1 : float
        L1 regularization factor.
    l2 : float
        L2 regularization factor.

    """
    def __init__(self, l1=0., l2=0.):
        self.l1, self.l2 = l1, l2

    def __call__(self, x):
        pass


# Aliases.
def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)
