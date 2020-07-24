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
#     <https://github.com/pytorch/pytorch/blob/master/torch/nn/_reduction.py>
#
# ------------------------------------------------------------
"""Utilities for loss reduction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import logging


def legacy_get_string(size_average, reduce, emit_warning=True):
    warning = "size_average and reduce args will be deprecated," \
              " please use reduction='{}' instead."
    size_average = True if size_average is None else size_average
    reduce = True if reduce is None else reduce
    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        logging.warning(warning.format(ret))
    return ret
