# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/utils/losses_utils.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.tensorflow.core.ops.losses import loss_reduction


Reduction = loss_reduction.Reduction