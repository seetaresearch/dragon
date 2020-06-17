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

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

from dragon.vm.caffe.net import Net
from dragon.vm.caffe.net_spec import layers
from dragon.vm.caffe.net_spec import NetSpec
from dragon.vm.caffe.net_spec import params
from dragon.vm.caffe.net_spec import to_proto
from dragon.vm.caffe.solver import AdamSolver
from dragon.vm.caffe.solver import NesterovSolver
from dragon.vm.caffe.solver import RMSPropSolver
from dragon.vm.caffe.solver import SGDSolver

# Aliases
Layer = object
TRAIN = 'TRAIN'
TEST = 'TEST'
