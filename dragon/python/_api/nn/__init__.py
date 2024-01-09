# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""NN API."""

from dragon.core.ops.rnn_ops import GRU
from dragon.core.ops.rnn_ops import LSTM
from dragon.core.ops.rnn_ops import RNN
from dragon.core.ops.activation_ops import dropout
from dragon.core.ops.activation_ops import drop_block
from dragon.core.ops.activation_ops import drop_path
from dragon.core.ops.activation_ops import elu
from dragon.core.ops.activation_ops import gelu
from dragon.core.ops.activation_ops import hardsigmoid
from dragon.core.ops.activation_ops import hardswish
from dragon.core.ops.activation_ops import leaky_relu
from dragon.core.ops.activation_ops import log_softmax
from dragon.core.ops.activation_ops import prelu
from dragon.core.ops.activation_ops import relu
from dragon.core.ops.activation_ops import relu6
from dragon.core.ops.activation_ops import selu
from dragon.core.ops.activation_ops import silu
from dragon.core.ops.activation_ops import softmax
from dragon.core.ops.array_ops import channel_shuffle
from dragon.core.ops.math_ops import moments
from dragon.core.ops.normalization_ops import batch_norm
from dragon.core.ops.normalization_ops import channel_norm
from dragon.core.ops.normalization_ops import group_norm
from dragon.core.ops.normalization_ops import instance_norm
from dragon.core.ops.normalization_ops import layer_norm
from dragon.core.ops.normalization_ops import local_response_norm
from dragon.core.ops.normalization_ops import lp_norm
from dragon.core.ops.normalization_ops import sync_batch_norm
from dragon.core.ops.vision_ops import bias_add
from dragon.core.ops.vision_ops import conv
from dragon.core.ops.vision_ops import conv_transpose
from dragon.core.ops.vision_ops import conv1d
from dragon.core.ops.vision_ops import conv1d_transpose
from dragon.core.ops.vision_ops import conv2d
from dragon.core.ops.vision_ops import conv2d_transpose
from dragon.core.ops.vision_ops import conv3d
from dragon.core.ops.vision_ops import conv3d_transpose
from dragon.core.ops.vision_ops import depth_to_space
from dragon.core.ops.vision_ops import pool
from dragon.core.ops.vision_ops import pool1d
from dragon.core.ops.vision_ops import pool2d
from dragon.core.ops.vision_ops import pool3d
from dragon.core.ops.vision_ops import space_to_depth
