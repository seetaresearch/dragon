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

from dragon.vm.tensorflow.core.ops.array_ops import depth_to_space
from dragon.vm.tensorflow.core.ops.array_ops import space_to_depth
from dragon.vm.tensorflow.core.ops.nn_ops import avg_pool
from dragon.vm.tensorflow.core.ops.nn_ops import avg_pool1d
from dragon.vm.tensorflow.core.ops.nn_ops import avg_pool2d
from dragon.vm.tensorflow.core.ops.nn_ops import avg_pool3d
from dragon.vm.tensorflow.core.ops.nn_ops import bias_add
from dragon.vm.tensorflow.core.ops.nn_ops import conv1d
from dragon.vm.tensorflow.core.ops.nn_ops import conv2d
from dragon.vm.tensorflow.core.ops.nn_ops import conv3d
from dragon.vm.tensorflow.core.ops.nn_ops import conv1d_transpose
from dragon.vm.tensorflow.core.ops.nn_ops import conv2d_transpose
from dragon.vm.tensorflow.core.ops.nn_ops import conv3d_transpose
from dragon.vm.tensorflow.core.ops.nn_ops import convolution
from dragon.vm.tensorflow.core.ops.nn_ops import conv_transpose
from dragon.vm.tensorflow.core.ops.nn_ops import dropout
from dragon.vm.tensorflow.core.ops.nn_ops import elu
from dragon.vm.tensorflow.core.ops.nn_ops import fused_batch_norm
from dragon.vm.tensorflow.core.ops.nn_ops import gelu
from dragon.vm.tensorflow.core.ops.nn_ops import l2_loss
from dragon.vm.tensorflow.core.ops.nn_ops import l2_normalize
from dragon.vm.tensorflow.core.ops.nn_ops import leaky_relu
from dragon.vm.tensorflow.core.ops.nn_ops import local_response_normalization
from dragon.vm.tensorflow.core.ops.nn_ops import log_softmax
from dragon.vm.tensorflow.core.ops.nn_ops import max_pool
from dragon.vm.tensorflow.core.ops.nn_ops import max_pool1d
from dragon.vm.tensorflow.core.ops.nn_ops import max_pool2d
from dragon.vm.tensorflow.core.ops.nn_ops import max_pool3d
from dragon.vm.tensorflow.core.ops.nn_ops import moments
from dragon.vm.tensorflow.core.ops.nn_ops import relu
from dragon.vm.tensorflow.core.ops.nn_ops import relu6
from dragon.vm.tensorflow.core.ops.nn_ops import selu
from dragon.vm.tensorflow.core.ops.nn_ops import sigmoid_cross_entropy_with_logits
from dragon.vm.tensorflow.core.ops.nn_ops import silu
from dragon.vm.tensorflow.core.ops.nn_ops import softmax
from dragon.vm.tensorflow.core.ops.nn_ops import softmax_cross_entropy_with_logits
from dragon.vm.tensorflow.core.ops.nn_ops import sparse_softmax_cross_entropy_with_logits
