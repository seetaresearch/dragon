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

import dragon.core.mpi as mpi

from . import *


@OpSchema.Inputs(1)
def MPIBroadcast(inputs, root, mpi_ranks=None, **kwargs):
    """Broadcast a tensor to all nodes in the ``MPIGroup``.

    Parameters
    ----------
    inputs : Tensor
        The tensor to broadcast.
    root : int
        The world rank of root node.
    mpi_ranks: sequence of int, optional
        The world rank of nodes in group. Default is ``None`` (Use All).

    Returns
    -------
    Tensor
        The broadcast output.

    Notes
    -----
    For root, the output **shares** the input.

    For others, the input is **inaccessible**.

    """
    arguments = ParseArgs(locals())
    if mpi_ranks is None:
        num_nodes = mpi.Size()
        mpi_ranks = [i for i in range(0, num_nodes)]
    if not isinstance(mpi_ranks, list): mpi_rank = [mpi_ranks]

    comm, group = mpi.CreateGroup(root, incl=mpi_ranks)
    arguments = {'inputs': arguments['inputs'], 'comm': comm, 'group': group}
    return Tensor.CreateOperator('MPIBroadcast', **arguments)


@OpSchema.Inputs(1)
def MPIGather(inputs, root, mpi_ranks=None, **kwargs):
    """Gather a tensor from all nodes to root in the ``MPIGroup``.

    Parameters
    ----------
    inputs : Tensor
        The tensor to gather.
    root : int
        The world rank of root node.
    mpi_ranks: sequence of int, optional
        The world rank of nodes in group. Default is ``None`` (Use All).

    Returns
    -------
    sequence of Tensor
        The gathered outputs.

    Notes
    -----
    The number of outputs is decided on the number of ``mpi_ranks``.

    The outputs are **accessible** only for root and vice versa.

    """
    arguments = ParseArgs(locals())

    if mpi_ranks is None:
        num_nodes = mpi.Size()
        mpi_ranks = [i for i in range(0, num_nodes)]
    if not isinstance(mpi_ranks, list): mpi_ranks = [mpi_ranks]

    comm, group = mpi.CreateGroup(root, incl=mpi_ranks)

    arguments = {
        'inputs': arguments['inputs'],
        'comm': comm,
        'group': group,
        'num_outputs': len(mpi_ranks)
    }

    return Tensor.CreateOperator('MPIGather', **arguments)