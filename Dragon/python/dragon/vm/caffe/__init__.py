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

"""Import extended PyCaffe modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Solver
from dragon.vm.caffe.solver import (
    SGDSolver,
    NesterovSolver,
    RMSPropSolver,
    AdamSolver,
)

# Net
from dragon.vm.caffe.net import (
    Net,
    PartialNet,
)

# Singleton
from dragon.vm.caffe.misc import (
    set_mode_cpu,
    set_mode_gpu,
    set_device,
    set_random_seed,
    root_solver,
    set_root_solver,
)

# NetSpec
from dragon.vm.caffe.net_spec import (
    layers,
    params,
    NetSpec,
    to_proto,
)

# Phase
TRAIN = "TRAIN"
TEST = "TEST"

# Alias
Layer = object