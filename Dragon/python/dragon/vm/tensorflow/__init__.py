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

# Framework
from dragon.vm.tensorflow.framework.framework_lib import *

# Session
from dragon.vm.tensorflow.client.client_lib import *

# Ops
from dragon.vm.tensorflow.ops.standard_ops import *

# Bring in subpackages.
from dragon.vm.tensorflow.ops import nn

# Import the names from training.py as train.Name.
from dragon.vm.tensorflow.training import training as train

# Export modules and constants
from dragon.vm.tensorflow import keras
from dragon.vm.tensorflow.layers import layers
from dragon.vm.tensorflow.ops import losses

# Make some application and test modules available.
from dragon.vm.tensorflow.platform import tf_logging as logging
