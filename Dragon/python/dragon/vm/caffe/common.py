# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.config as config

_root_solver = True

def set_mode_cpu():
    config.EnableCPU()

def set_mode_gpu():
    config.EnableCUDA()

def set_device(device):
    config.SetGPU(device)

def set_random_seed(seed):
    config.SetRandomSeed(seed)

def root_solver():
    global _root_solver
    return _root_solver

def set_root_solver(val):
    global _root_solver
    _root_solver = val