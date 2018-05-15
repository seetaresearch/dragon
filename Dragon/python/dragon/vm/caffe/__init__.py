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

from .solver import SGDSolver, NesterovSolver, RMSPropSolver, AdamSolver
from .net import Net, PartialNet
from .misc import set_mode_cpu, set_mode_gpu, set_device, set_random_seed, \
    root_solver, set_root_solver

Layer = object
TRAIN = "TRAIN"
TEST = "TEST"

from .net_spec import layers, params, NetSpec, to_proto