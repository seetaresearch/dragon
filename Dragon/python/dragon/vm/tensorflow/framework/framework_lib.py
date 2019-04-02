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

# The Graph (Workspace:))
from dragon.core.workspace import Workspace as Graph

# Utilities used when building a Graph
from dragon.vm.tensorflow.framework.ops import device
from dragon.vm.tensorflow.framework.ops import name_scope
from dragon.vm.tensorflow.framework.ops import get_default_graph
from dragon.vm.tensorflow.framework.ops import reset_default_graph
from dragon.vm.tensorflow.framework.ops import add_to_collection
from dragon.vm.tensorflow.framework.ops import get_collection
from dragon.vm.tensorflow.framework.ops import convert_to_tensor
from dragon.vm.tensorflow.framework.ops import GraphKeys
from dragon.vm.tensorflow.framework.constant_op import *
from dragon.vm.tensorflow.framework.dtypes import *

# Utilities used to represent a Tensor
from dragon.vm.tensorflow.framework.tensor_shape import Dimension
from dragon.vm.tensorflow.framework.tensor_shape import TensorShape