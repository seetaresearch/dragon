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
"""NN functional module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

from dragon.vm.torch.core.nn.functional import adaptive_avg_pool1d
from dragon.vm.torch.core.nn.functional import adaptive_avg_pool2d
from dragon.vm.torch.core.nn.functional import adaptive_avg_pool3d
from dragon.vm.torch.core.nn.functional import adaptive_max_pool1d
from dragon.vm.torch.core.nn.functional import adaptive_max_pool2d
from dragon.vm.torch.core.nn.functional import adaptive_max_pool3d
from dragon.vm.torch.core.nn.functional import affine
from dragon.vm.torch.core.nn.functional import avg_pool1d
from dragon.vm.torch.core.nn.functional import avg_pool2d
from dragon.vm.torch.core.nn.functional import avg_pool3d
from dragon.vm.torch.core.nn.functional import batch_norm
from dragon.vm.torch.core.nn.functional import binary_cross_entropy_with_logits
from dragon.vm.torch.core.nn.functional import channel_norm
from dragon.vm.torch.core.nn.functional import channel_shuffle
from dragon.vm.torch.core.nn.functional import conv1d
from dragon.vm.torch.core.nn.functional import conv2d
from dragon.vm.torch.core.nn.functional import conv3d
from dragon.vm.torch.core.nn.functional import conv_transpose1d
from dragon.vm.torch.core.nn.functional import conv_transpose2d
from dragon.vm.torch.core.nn.functional import conv_transpose3d
from dragon.vm.torch.core.nn.functional import cosine_similarity
from dragon.vm.torch.core.nn.functional import cross_entropy
from dragon.vm.torch.core.nn.functional import ctc_loss
from dragon.vm.torch.core.nn.functional import depthwise_conv2d
from dragon.vm.torch.core.nn.functional import drop_block2d
from dragon.vm.torch.core.nn.functional import drop_path
from dragon.vm.torch.core.nn.functional import dropout
from dragon.vm.torch.core.nn.functional import elu
from dragon.vm.torch.core.nn.functional import embedding
from dragon.vm.torch.core.nn.functional import gelu
from dragon.vm.torch.core.nn.functional import group_norm
from dragon.vm.torch.core.nn.functional import hardsigmoid
from dragon.vm.torch.core.nn.functional import hardswish
from dragon.vm.torch.core.nn.functional import kl_div
from dragon.vm.torch.core.nn.functional import l1_loss
from dragon.vm.torch.core.nn.functional import layer_norm
from dragon.vm.torch.core.nn.functional import leaky_relu
from dragon.vm.torch.core.nn.functional import linear
from dragon.vm.torch.core.nn.functional import local_response_norm
from dragon.vm.torch.core.nn.functional import log_softmax
from dragon.vm.torch.core.nn.functional import interpolate
from dragon.vm.torch.core.nn.functional import max_pool1d
from dragon.vm.torch.core.nn.functional import max_pool2d
from dragon.vm.torch.core.nn.functional import max_pool3d
from dragon.vm.torch.core.nn.functional import mse_loss
from dragon.vm.torch.core.nn.functional import multi_head_attention_forward
from dragon.vm.torch.core.nn.functional import nll_loss
from dragon.vm.torch.core.nn.functional import normalize
from dragon.vm.torch.core.nn.functional import one_hot
from dragon.vm.torch.core.nn.functional import pad
from dragon.vm.torch.core.nn.functional import pixel_shuffle
from dragon.vm.torch.core.nn.functional import pixel_unshuffle
from dragon.vm.torch.core.nn.functional import prelu
from dragon.vm.torch.core.nn.functional import relu
from dragon.vm.torch.core.nn.functional import relu6
from dragon.vm.torch.core.nn.functional import selu
from dragon.vm.torch.core.nn.functional import sigmoid
from dragon.vm.torch.core.nn.functional import sigmoid_focal_loss
from dragon.vm.torch.core.nn.functional import silu
from dragon.vm.torch.core.nn.functional import smooth_l1_loss
from dragon.vm.torch.core.nn.functional import softmax
from dragon.vm.torch.core.nn.functional import sync_batch_norm
from dragon.vm.torch.core.nn.functional import tanh
from dragon.vm.torch.core.nn.functional import unfold
from dragon.vm.torch.core.nn.functional import upsample
from dragon.vm.torch.core.nn.functional import upsample_bilinear
from dragon.vm.torch.core.nn.functional import upsample_nearest

__all__ = [_s for _s in dir() if not _s.startswith('_')]
