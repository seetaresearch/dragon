# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

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

