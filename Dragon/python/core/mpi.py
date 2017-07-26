# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon import MPIInitCC, MPIRankCC, MPISizeCC, MPICreateGroupCC, MPIFinalizeCC
import numpy as np

_is_init = False
_snapshot_ranks = []
_parallel_groups = []
_parallel_mode = 'Sync'

def init():
    MPIInitCC()
    global _is_init
    global _snapshot_ranks
    _is_init = True
    _snapshot_ranks = [i for i in xrange(size())]

def check_init():
    global _is_init
    if _is_init is False: init()

def is_init():
    return _is_init

def rank():
    check_init()
    return MPIRankCC()

def size():
    check_init()
    return MPISizeCC()

def group(root=0, incl=[], excl=[]):
    check_init()
    comm, group = MPICreateGroupCC(root, incl, excl)
    return np.int64(comm), np.int64(group)

def snapshot(incl):
    check_init()
    if not isinstance(incl, list): incl = [incl]
    global _snapshot_ranks
    _snapshot_ranks = incl

def parallel(conf):
    check_init()
    if not isinstance(conf[0], list): conf = [conf]
    for ele in conf:
        if not isinstance(ele, list):
            raise TypeError('parallel groups must be a list')
    global _parallel_groups
    _parallel_groups = conf

def allow_snapshot():
    global _snapshot_ranks
    return rank() in _snapshot_ranks

def allow_parallel():
    global _parallel_groups
    world_rank = rank()
    for idx, g in enumerate(_parallel_groups):
        if world_rank in g: return idx, g
    return -1, []

def set_parallel_mode(mode):
    assert mode == 'Sync' or \
           mode == 'Async' \
           or mode == 'Async_No_Lock'
    global _parallel_mode
    _parallel_mode = mode

def get_parallel_mode():
    global _parallel_mode
    return _parallel_mode

def finalize():
    check_init()
    MPIFinalizeCC()