# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/pytorch/pytorch/blob/master/torch/distributed/distributed_c10d.py>
#
# ------------------------------------------------------------
"""Distributed utilities equipped with Python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import backend as _b
from dragon.core.util import nest
from dragon.core.util import six
from dragon.core.util import tls


class Backend(object):
    """An enum-like class of available backends: MPI and NCCL."""

    UNDEFINED = 'UNDEFINED'
    AUTO = 'AUTO'
    MPI = 'MPI'
    NCCL = 'NCCL'

    def __new__(cls, name):
        if not isinstance(name, six.string_types):
            raise ValueError('Backend name must be a string, but got: {}'.format(name))
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)
        if value == 'AUTO':
            if _b.cudaIsNCCLSufficient():
                return Backend.NCCL
            return Backend.MPI
        elif value == 'NCCL':
            if not _b.cudaIsNCCLSufficient():
                raise ValueError('NCCL backend is not available.')
        elif value == Backend.UNDEFINED:
            raise ValueError("Invalid backend: '{}'".format(name))
        return value


class ProcessGroup(object):
    """A group that stores a set of ranks."""

    def __init__(self, ranks, comm, handle, backend):
        self._handle = handle
        self._ranks, self._comm = ranks, comm
        if backend is None:
            self._backend = Backend('AUTO')
        else:
            self._backend = Backend(backend)
        self._arguments = {
            # Stored for executing the collective ops.
            'comm': self._comm, 'group': self._handle,
            'backend': self._backend, 'ranks': self._ranks,
        }

    @property
    def arguments(self):
        """Return the arguments of this group.

        Returns
        -------
        dict
            The arguments.

        """
        return self._arguments

    @property
    def backend(self):
        """Return the backend of this group.

        Returns
        -------
        str
            The backend spec.

        """
        return self._backend

    @property
    def ranks(self):
        """Return the ranks of this group.

        Returns
        -------
        Sequence[int]
            The ranks.

        """
        return self._ranks

    @property
    def size(self):
        """Return the size of this group.

        Returns
        -------
        int
            The group size.

        """
        return len(self._ranks) if self._ranks is not None else 0

    def as_default(self):
        """Switch ``self`` as the default process group.

        Call this method with the **with** keyword.

        Once **with** is exited, the previous default will be set.

        Returns
        -------
        ProcessGroup
            The ``self``.

        """
        if self._ranks is None:
            return _GLOBAL_PROCESS_GROUP_STACK.get_controller(None)
        return _GLOBAL_PROCESS_GROUP_STACK.get_controller(self)

    def __repr__(self):
        return '%s:%d' % (self._backend, self._handle)


def init():
    """Init the distributed env."""
    if is_initialized():
        # ATTENTION: MPI env can only be initialized once per process.
        return
    _b.MPIInit()
    global _GLOBAL_MPI_CONTEXT
    _GLOBAL_MPI_CONTEXT = _MPIContext()


def is_initialized():
    """Whether the distributed env has initialized.

    Returns
    -------
    bool
        **True** if env has initialized otherwise **False**.

    """
    return _GLOBAL_MPI_CONTEXT is not None


def get_backend(group):
    """Return the backend of given process group.

    Parameters
    ----------
    group : ProcessGroup
        The process group to query backend.

    Returns
    -------
    str
        The backend spec.

    """
    return group.backend


def get_group():
    """Return the current default process group.

    Returns
    -------
    ProcessGroup
        The default group.

    """
    return _GLOBAL_PROCESS_GROUP_STACK.get_default()


def get_rank(group=None):
    """Return the rank of current process.

    If ``group`` is **None**, return the world rank.
    Otherwise, return the rank in the group.

    Parameters
    ----------
    group : ProcessGroup, optional
        The optional process group.

    Returns
    -------
    int
        The rank.

    """
    init()
    world_rank = _b.MPIRank()
    if group is not None:
        for idx, rank in enumerate(group.ranks):
            if rank == world_rank:
                return idx
    return world_rank


def get_world_size():
    """Return the world size of distributed env.

    Returns
    -------
    int
        The world size.

    """
    init()
    return _b.MPISize()


def new_group(ranks=None, backend=None, verbose=False):
    """Create a new communication group.

    The ``ranks`` can be set to **None** to create
    an empty group for disabling the distributed utilities.

    If ``backend`` is **None**, select as: **NCCL** > **MPI**.

    Note that this function should be called from all processes,
    even if they are not going to be included in this group.

    Parameters
    ----------
    ranks : Sequence[int], optional
        The rank of processes to be included.
    backend : {'AUTO', 'MPI', 'NCCL'}, optional
        The optional backend.
    verbose : bool, optional, default=False
        **True** to log the group info.

    """
    if ranks is None:
        return ProcessGroup(None, None, None, backend)
    else:
        init()
        ranks = nest.flatten(ranks)
        comm, handle = _b.MPICreateGroup(ranks, verbose)
        return ProcessGroup(ranks, comm, handle, backend)


class _MPIContext(object):
    """Context to finalize mpi under destruction."""

    def __del__(self):
        _b.MPIFinalize()


_GLOBAL_MPI_CONTEXT = None
_GLOBAL_PROCESS_GROUP_STACK = tls.Stack()
