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

from dragon.vm.tensorflow.training.optimizer import (
    GradientDescentOptimizer,
    MomentumOptimizer,
    RMSPropOptimizer,
    AdamOptimizer,
)

from dragon.vm.tensorflow.training.learning_rate_decay import (
    piecewise_constant,
    piecewise_constant_decay,
    exponential_decay,
    natural_exp_decay,
    cosine_decay,
    cosine_decay_restarts,
)