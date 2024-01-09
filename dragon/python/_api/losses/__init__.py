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
"""Loss API."""

from dragon.core.ops.loss_ops import ctc_loss
from dragon.core.ops.loss_ops import l1_loss
from dragon.core.ops.loss_ops import l2_loss
from dragon.core.ops.loss_ops import nll_loss
from dragon.core.ops.loss_ops import sigmoid_cross_entropy_loss
from dragon.core.ops.loss_ops import sigmoid_focal_loss
from dragon.core.ops.loss_ops import smooth_l1_loss
from dragon.core.ops.loss_ops import softmax_cross_entropy_loss
