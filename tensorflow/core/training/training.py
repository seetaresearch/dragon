# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# flake8: noqa
from dragon.vm.tensorflow.core.training.learning_rate_decay import cosine_decay
from dragon.vm.tensorflow.core.training.learning_rate_decay import cosine_decay_restarts
from dragon.vm.tensorflow.core.training.learning_rate_decay import exponential_decay
from dragon.vm.tensorflow.core.training.learning_rate_decay import natural_exp_decay
from dragon.vm.tensorflow.core.training.learning_rate_decay import piecewise_constant
from dragon.vm.tensorflow.core.training.learning_rate_decay import piecewise_constant_decay
from dragon.vm.tensorflow.core.training.optimizer import GradientDescentOptimizer
from dragon.vm.tensorflow.core.training.optimizer import MomentumOptimizer
from dragon.vm.tensorflow.core.training.optimizer import RMSPropOptimizer
from dragon.vm.tensorflow.core.training.optimizer import AdamOptimizer
