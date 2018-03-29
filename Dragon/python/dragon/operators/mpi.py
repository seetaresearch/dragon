# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
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


def MPIBroadcast(inputs, root, mpi_ranks=None, **kwargs):
    """Broadcast a tensor to all nodes in the ``MPIGroup``.

    Parameters
    ----------
    inputs : Tensor
        The tensor to broadcast.
    root : int
        The world rank of root node.
    mpi_ranks: int, list of int or None
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
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
    if mpi_ranks is None:
        num_nodes = mpi.Size()
        mpi_ranks = [i for i in range(0, num_nodes)]
    if not isinstance(mpi_ranks, list): mpi_rank = [mpi_ranks]

    comm, group = mpi.CreateGroup(root, incl=mpi_ranks)
    arguments = {'inputs': arguments['inputs'], 'comm': comm, 'group': group}

    output = Tensor.CreateOperator(nout=1, op_type='MPIBroadcast', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def MPIGather(inputs, root, mpi_ranks=None, **kwargs):
    """Gather a tensor from all nodes to root in the ``MPIGroup``.

    Parameters
    ----------
    inputs : Tensor
        The tensor to gather.
    root : int
        The world rank of root node.
    mpi_ranks: int, list of int or None
        The world rank of nodes in group. Default is ``None`` (Use All).

    Returns
    -------
    Tensor or list of Tensor
        The gathered outputs.

    Notes
    -----
    The number of outputs is decided on the number of ``mpi_ranks``.

    The outputs are **accessible** only for root and vice versa.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    if mpi_ranks is None:
        num_nodes = mpi.Size()
        mpi_ranks = [i for i in range(0, num_nodes)]
    if not isinstance(mpi_ranks, list): mpi_ranks = [mpi_ranks]

    comm, group = mpi.CreateGroup(root, incl=mpi_ranks)
    arguments = {'inputs': arguments['inputs'], 'comm': comm, 'group': group}

    outputs = Tensor.CreateOperator(nout=len(mpi_ranks), op_type='MPIGather', **arguments)

    if inputs.shape is not None:
        if isinstance(outputs, list):
            for output in outputs:
                output.shape = inputs.shape[:]
        else: outputs.shape = inputs.shape[:]

    return outputs