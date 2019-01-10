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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import dragon.import_c_api as C


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
    'Finalize',
]


_GLOBAL_MPI_IS_INIT = False
_GLOBAL_MPI_SNAPSHOT_RANKS = []
_GLOBAL_MPI_PARALLEL_GROUPS = []
_GLOBAL_MPI_PARALLEL_MODE = 'MPI'


def _check_init():
    if _GLOBAL_MPI_IS_INIT is False: Init()


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
    C.MPIInitCC()
    global _GLOBAL_MPI_IS_INIT
    global _GLOBAL_MPI_SNAPSHOT_RANKS
    _GLOBAL_MPI_IS_INIT = True
    _GLOBAL_MPI_SNAPSHOT_RANKS = [i for i in range(Size())]


def Is_Init():
    """Whether the MPI env has initialized.

    Returns
    -------
    boolean

    """
    return _GLOBAL_MPI_IS_INIT


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
    return C.MPIRankCC()


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
    return C.MPISizeCC()


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
    comm, group = C.MPICreateGroupCC(root, incl, excl)
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
    global _GLOBAL_MPI_SNAPSHOT_RANKS
    _GLOBAL_MPI_SNAPSHOT_RANKS = incl


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
    >>> import dragon.core.mpi as mpi
    >>> mpi.Parallel([0, 1]) # rank(0, 1) will be into a parallel group.
    >>> mpi.Parallel([0, 1], [2, 3]) # rank(0, 1), rank(2, 3) will be into two parallel groups.

    """
    _check_init()
    if not isinstance(conf[0], list): conf = [conf]
    for ele in conf:
        if not isinstance(ele, list):
            raise TypeError('parallel groups must be a list')
    global _GLOBAL_MPI_PARALLEL_GROUPS
    _GLOBAL_MPI_PARALLEL_GROUPS = conf


def AllowSnapshot():
    """Whether this node can snapshot.

    Returns
    -------
    boolean
    """
    return Rank() in _GLOBAL_MPI_SNAPSHOT_RANKS


def AllowParallel():
    """Whether this node was set for data parallelism.

    Returns
    -------
    boolean

    """
    world_rank = Rank()
    for idx, g in enumerate(_GLOBAL_MPI_PARALLEL_GROUPS):
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
           mode == 'NCCL' or \
           mode == 'MIXED'
    global _GLOBAL_MPI_PARALLEL_MODE
    _GLOBAL_MPI_PARALLEL_MODE = mode


def GetParallelMode():
    """Get the current mode of data parallelism.

    Returns
    -------
    str
        The mode, ``MPI``, ``NCCL`` or ``MIXED``.

    """
    return _GLOBAL_MPI_PARALLEL_MODE


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
    C.MPIFinalizeCC()