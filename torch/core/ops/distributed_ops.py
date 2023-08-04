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
    """Gather tensor across all nodes and output to a tensor list.

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
    Sequence[dragon.vm.torch.Tensor]
        The output tensor list.

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
    return [output_tensor]


def all_gather_into_tensor(output_tensor, input_tensor, group=None):
    """Gather tensor across all nodes and output to a tensor.

    Parameters
    ----------
    output_tensor : dragon.vm.torch.Tensor
        The output tensor.
    input_tensor : dragon.vm.torch.Tensor
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
        return input_tensor
    return Function.apply(
        'Collective', input_tensor.device, [input_tensor],
        outputs=[output_tensor], operation='ALLGATHER', **group.arguments)


def all_reduce(tensor, op='sum', group=None):
    """Reduce tensor across all nodes.

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
    """Broadcast tensor from the source node.

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


def reduce_scatter(output, input_list, op='sum', group=None):
    """Reduce and scatter the tensor list across all nodes.

    Parameters
    ----------
    output : dragon.vm.torch.Tensor
        The output tensor.
    input_list : Sequence[dragon.vm.torch.Tensor]
        The input tensor list.
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
        return input_list
    op = op.upper()
    if op not in ('SUM',):
        raise ValueError('Unsupported reduction: ' + op)
    if len(input_list) > 0:
        input = Function.apply('Concat', input_list[0].device, input_list)
    else:
        input = input_list[0]
    return Function.apply(
        'Collective', input.device, [input], outputs=[output],
        operation='REDUCESCATTER', reduction=op, **group.arguments)


def reduce_scatter_tensor(output, input, op='sum', group=None):
    """Reduce and scatter the tensor across all nodes.

    Parameters
    ----------
    output : dragon.vm.torch.Tensor
        The output tensor.
    input : dragon.vm.torch.Tensor
        The input tensor.
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
        return input
    op = op.upper()
    if op not in ('SUM',):
        raise ValueError('Unsupported reduction: ' + op)
    return Function.apply(
        'Collective', input.device, [input], outputs=[output],
        operation='REDUCESCATTER', reduction=op, **group.arguments)
