# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""A fast open framework for deep learning."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

# Classes
from dragon.vm.caffe.core.net import Net
from dragon.vm.caffe.core.net_spec import NetSpec
from dragon.vm.caffe.core.solver import AdamSolver
from dragon.vm.caffe.core.solver import NesterovSolver
from dragon.vm.caffe.core.solver import RMSPropSolver
from dragon.vm.caffe.core.solver import SGDSolver
from dragon.vm.caffe.core.solver import Solver

# Functions
from dragon.vm.caffe.core.net_spec import to_proto

# Attributes
from dragon.vm.caffe.core.net_spec import layers
from dragon.vm.caffe.core.net_spec import params
__all__ = [_s for _s in dir() if not _s.startswith('_')]

# Aliases
Layer = object
TRAIN = 'TRAIN'
TEST = 'TEST'
