# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
from __init__ import *

option = {}

REGISTERED_OPERATORS = set(s for s in RegisteredOperatorsCC())
NO_GRADIENT_OPERATORS = set(s for s in NoGradientOperatorsCC())

option['device'] = 'CPU'
option['gpu_id'] = 0
option['use_cudnn'] = False
option['random_seed'] = 3
option['debug_mode'] = True

def EnableCPU():
    global option
    option['device'] = 'CPU'

def EnableCUDA(gpu_id=0, use_cudnn=True):
    global option
    option['device'] = 'CUDA'
    option['gpu_id'] = gpu_id
    option['use_cudnn'] = use_cudnn

# TODO(Pan): please not use @setter
# TODO(Pan): seems that it can't change the global value

def SetRandomSeed(seed):
    global option
    option['random_seed'] = seed

def GetRandomSeed():
    global option
    return option['random_seed']

def SetGPU(id):
    global option
    option['gpu_id'] = id

def GetGPU(id):
    global option
    return option['gpu_id']

def SetDebugMode(mode):
    global option
    option['debug_mode'] = mode

def SetLoggingLevel(level):
    """
    set the minimum level of logging
    :param level:  a str of DEBUG, INFO(default), WARNING, ERROR, FATAL
    """

    SetLogLevelCC(level)





