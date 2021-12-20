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
from dragon.vm.torch.core.autograd.function import Function


def all_gather(tensor_list, tensor, group=None):
    """Gather the tensor across all nodes in a group.

    Parameters
    ----------
    tensor_list : Sequence[dragon.vm.torch.Tensor]
        The output tensor list.
    tensor : dragon.vm.torch.Tensor
        The tensor to be sent.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    group = group or distributed.get_group()
    if group is None:
        return tensor
    output_tensor = Function.apply(
        'Collective', tensor.device, [tensor],
        operation='ALLGATHER', **group.arguments)
    if len(tensor_list) > 0:
        return Function.apply(
            'Split', output_tensor.device, [output_tensor],
            outputs=[None] * len(tensor_list),
            axis=0, size_split=None, copy=True)
    return output_tensor


def all_reduce(tensor, op='sum', group=None):
    """Reduce the tensor across all nodes in a group.

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The tensor to reduce.
    op : str, optional
        The reduction op.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    group = group or distributed.get_group()
    if group is None:
        return tensor
    op = op.upper()
    if op not in ('MEAN', 'SUM'):
        raise ValueError('Unsupported reduction: ' + op)
    return Function.apply(
        'Collective', tensor.device, [tensor], outputs=[tensor],
        operation='ALLREDUCE', reduction=op, **group.arguments)


def broadcast(tensor, src=0, group=None):
    """Broadcast the tensor from source node in a group.

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The tensor to be sent.
    src : int
        The rank of the source node.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    group = group or distributed.get_group()
    if group is None:
        return tensor
    return Function.apply(
        'Collective', tensor.device, [tensor], outputs=[tensor],
        operation='BROADCAST', root=src, **group.arguments)
