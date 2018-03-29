# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

# config
from dragon.config import *
import dragon.config as config

# core
from dragon.core.tensor import Tensor
import dragon.core.workspace as workspace

# ops
from dragon.ops import *

# updaters
from dragon.updaters import *

# theano utilities
from dragon.vm.theano.compile.function import function as function
from dragon.vm.theano.tensor import grad as grad

# scope
from dragon.core.scope import TensorScope as name_scope
from dragon.core.scope import PhaseScope as phase_scope
from dragon.core.scope import DeviceScope as device_scope

