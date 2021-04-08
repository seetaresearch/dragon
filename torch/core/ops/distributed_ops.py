# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Distributed ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core import distributed
from dragon.core.util import nest
from dragon.vm.torch.core.autograd.function_impl import FunctionLib


def all_reduce(tensor, op='SUM', group=None):
    """Reduce the tensor across all nodes in a group.

    Parameters
    ----------
    tensor : Sequence[dragon.vm.torch.Tensor]
        The tensor(s) to reduce.
    op : {'SUM', 'MEAN'}, optional
        The reduce operation.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if group is None:
        group = distributed.get_group()
    if group is None:
        raise ValueError('<group> is required.')
    if op not in ('MEAN', 'SUM'):
        raise ValueError('Unsupported reduce op: ' + op)
    tensors = nest.flatten(tensor)
    return FunctionLib.apply(
        'Collective', tensors[0].device, tensors, outputs=tensors,
        communication='ALLREDUCE', operation=op, **group.arguments)


def broadcast(tensor, src=0, group=None):
    """Broadcast the tensor from source node in a group.

    Parameters
    ----------
    tensor : Sequence[dragon.vm.torch.Tensor]
        The tensor(s) to reduce.
    src : int
        The rank of the source node.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if group is None:
        group = distributed.get_group()
    if group is None:
        raise ValueError('<group> is required.')
    tensors = nest.flatten(tensor)
    return FunctionLib.apply(
        'Collective', tensors[0].device, tensors, outputs=tensors,
        communication='BROADCAST', root=src, **group.arguments)
