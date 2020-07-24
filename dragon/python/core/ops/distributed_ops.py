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
from dragon.core.eager import context
from dragon.core.ops import distributed_ops_lib
from dragon.core.ops.utils import OpSchema
from dragon.core.ops.utils import parse_args


@OpSchema.num_inputs(1)
def all_reduce(inputs, operation='MEAN', group=None, **kwargs):
    """Reduce the input across all nodes in a group.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    operation : {'MEAN', 'SUM'}, optional
        The reduce operation.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    if group is None:
        group = distributed.get_group()
    if group is None:
        raise ValueError('<group> is required.')
    if operation not in ('MEAN', 'SUM'):
        raise ValueError('Unsupported reduce op:', operation)
    args.update(group.arguments)
    args.pop('group')
    op_lib = distributed_ops_lib.Collective
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                operation=operation,
                communication='ALLREDUCE',
                group=group,
            ).apply(inputs)
    else:
        return op_lib.blend(communication='ALLREDUCE', **args)


@OpSchema.num_inputs(1)
def broadcast(inputs, root=0, group=None, **kwargs):
    """Broadcast the input from root node in a group.

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor to broadcast.
    root : int, optional, default=0
        The node index in the group.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    if group is None:
        group = distributed.get_group()
    if group is None:
        raise ValueError('<group> is required.')
    args.update(group.arguments)
    args.pop('group')
    op_lib = distributed_ops_lib.Collective
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                root=root,
                communication='BROADCAST',
                group=group,
            ).apply(inputs)
    else:
        return op_lib.blend(communication='BROADCAST', **args)
