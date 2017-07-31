# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from .solver import SGDSolver, NesterovSolver, RMSPropSolver, AdamSolver
from .net import Net, PartialNet
from .common import set_mode_cpu, set_mode_gpu, set_device, set_random_seed, \
    root_solver, set_root_solver

Layer = object
TRAIN = "TRAIN"
TEST = "TEST"

from .net_spec import layers, params, NetSpec, to_proto