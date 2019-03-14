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

"""General Importing Portal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Config
from dragon.config import *
import dragon.config as config

# Core
from dragon.core.tensor import Tensor
import dragon.core.workspace as workspace
import dragon.core.tensor_utils as tensor_utils
import dragon.core.mpi as mpi
import dragon.core.cuda as cuda
import dragon.memonger as memonger

# Operators
from dragon.ops import *

# Updaters
from dragon.updaters import *

# Graph Primitives
from dragon.vm.theano.compile.function import function as function
from dragon.vm.theano.tensor import grad as grad

# Scopes
from dragon.core.scope import name_scope, get_default_name_scope
from dragon.core.scope import phase_scope, get_default_phase
from dragon.core.scope import device_scope, get_default_device
from dragon.core.scope import WorkspaceScope as ws_scope

# Version
from dragon.version import version
__version__ = version

# Logging
import dragon.core.logging as logging

