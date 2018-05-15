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

from dragon.vm.tensorflow.layers.convolutional import conv2d

from dragon.vm.tensorflow.layers.core import dense

from dragon.vm.tensorflow.layers.normalization import \
    batch_normalization, batch_norm, BatchNorm

from dragon.vm.tensorflow.layers.pooling import \
    average_pooling2d, max_pooling2d