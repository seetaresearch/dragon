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
#     <https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py>
#
# ------------------------------------------------------------
"""Module utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from dragon.core.util import six


def _get_adaptive_pool_args(input_sizes, output_sizes):
    stride, kernel_size = [], []
    for input_size, output_size in zip(input_sizes, output_sizes):
        if output_size == 1:
            stride.append(1)
            kernel_size.append(input_size)
        else:
            stride.append(input_size // output_size)
            kernel_size.append(input_size - (output_size - 1) * stride[-1])
    return {'kernel_size': kernel_size, 'stride': stride}


def _ntuple(n):
    def parse(x):
        if isinstance(x, six.collections_abc.Sequence):
            return x
        return tuple(itertools.repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
