# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range as xrange

from dragon.import_c_apis import *

_is_init = False
_snapshot_ranks = []
_parallel_groups = []
_parallel_mode = 'MPI'

__all__ = [
    'Init',
    'Is_Init',
    'Rank',
    'Size',
    'CreateGroup',
    'Snapshot',
    'AllowSnapshot',
    'Parallel',
    'AllowParallel',
    'SetParallelMode',
    'GetParallelMode',
    'Finalize'
]

def _check_init():
    global _is_init
    if _is_init is False: Init()

def Init():
    """Init the MPI env.

    Returns
    -------
    None

    Notes
    -----
    This function can only be called once.

    References
    ----------
    The wrapper of ``MPIInitCC``

    """
    MPIInitCC()
    global _is_init
    global _snapshot_ranks
    _is_init = True
    _snapshot_ranks = [i for i in xrange(Size())]


def Is_Init():
    """Whether the MPI env has initialized.

    Returns
    -------
    boolean

    """
    return _is_init


def Rank():
    """The world rank of current MPI node.

    Returns
    -------
    int
        The world rank.

    References
    ----------
    The wrapper of ``MPIRankCC``.

    """
    _check_init()
    return MPIRankCC()


def Size():
    """The world size of current MPI env.

    Returns
    -------
    int
        The world size.

    References
    ----------
    The wrapper of ``MPISizeCC``.

    """
    _check_init()
    return MPISizeCC()


def CreateGroup(root=0, incl=[], excl=[]):
    """Construct a ``MPIGroup`` with specific members.

    Parameters
    ----------
    root : int
        The root of this group.
    incl : list
        The include nodes.
    excl: list
        The exclude nodes.

    Returns
    -------
    tuple
        The local common and group id.

    References
    ----------
    The wrapper of ``MPICreateGroupCC``.

    """
    _check_init()
    comm, group = MPICreateGroupCC(root, incl, excl)
    return np.int64(comm), np.int64(group)


def Snapshot(incl):
    """Set the specific MPI nodes to snapshot.

    The exclude nodes will not snapshot through `workspace.Snapshot(*args, **kwargs)`_.

    Parameters
    ----------
    incl : int or list

    Returns
    -------
    None

    """
    _check_init()
    if not isinstance(incl, list): incl = [incl]
    global _snapshot_ranks
    _snapshot_ranks = incl


def Parallel(conf):
    """Set the specific MPI nodes for data parallelism.

    Parameters
    ----------
    conf : list
        The list of configures. Each configure should be a list also.

    Returns
    -------
    None

    Examples
    --------
    >>> mpi.parallel([0, 1]) # rank(0, 1) will be into a parallel group.

    >>> mpi.parallel([0, 1], [2, 3]) # rank(0, 1), rank(2, 3) will be into two parallel groups.

    """
    _check_init()
    if not isinstance(conf[0], list): conf = [conf]
    for ele in conf:
        if not isinstance(ele, list):
            raise TypeError('parallel groups must be a list')
    global _parallel_groups
    _parallel_groups = conf


def AllowSnapshot():
    """Whether this node can snapshot.

    Returns
    -------
    boolean
    """
    global _snapshot_ranks
    return Rank() in _snapshot_ranks


def AllowParallel():
    """Whether this node was set for data parallelism.

    Returns
    -------
    boolean

    """
    global _parallel_groups
    world_rank = Rank()
    for idx, g in enumerate(_parallel_groups):
        if world_rank in g: return idx, g
    return -1, []


def SetParallelMode(mode):
    """Set the mode of data parallelism.

    Parameters
    ----------
    mode : str
        The mode, ``MPI``, ``NCCL`` or ``MIXED``.

    Returns
    -------
    None

    Notes
    -----
    The default mode is ``MPI``.

    """
    assert mode == 'MPI' or \
           mode == 'NCCL' \
           or mode == 'MIXED'
    global _parallel_mode
    _parallel_mode = mode


def GetParallelMode():
    """Get the current mode of data parallelism.

    Returns
    -------
    str
        The mode, ``MPI``, ``NCCL`` or ``MIXED``.

    """
    global _parallel_mode
    return _parallel_mode


def Finalize():
    """Finalize the MPI env.

    Returns
    -------
    None

    Notes
    -----
    This function should be called to close the initialized MPI env.

    """
    _check_init()
    MPIFinalizeCC()