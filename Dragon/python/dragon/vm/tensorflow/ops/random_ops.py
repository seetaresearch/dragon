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

import dragon

from dragon.vm.tensorflow.framework import dtypes


__all__ = [
    'random_normal',
    'truncated_normal',
    'random_uniform'
]


def random_normal(
    shape, mean=0.0, stddev=1.0,
        dtype=dtypes.float32,
            seed=None, name=None):
    return dragon.ops.RandomNormal(shape, mean, stddev, name=name)


def truncated_normal(
    shape, mean=0.0, stddev=1.0,
        dtype=dtypes.float32,
            seed=None, name=None):
    return dragon.ops.TruncatedNormal(shape, mean, stddev, name=name)


def random_uniform(
    shape, minval=0, maxval=None,
        dtype=dtypes.float32,
            seed=None, name=None):
    return dragon.ops.RandomUniform(shape, minval, maxval, name=name)