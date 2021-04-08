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
from dragon.core.autograph import context
from dragon.core.autograph.op_impl import OpLib
from dragon.core.autograph.op_impl import OpSchema


@OpSchema.num_inputs(1)
def all_reduce(inputs, operation='mean', group=None, **kwargs):
    """Reduce the input across all nodes in a group.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    operation : {'mean', 'sum'}, optional
        The reduce operation.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    operation = operation.upper()
    if group is None:
        group = distributed.get_group()
    if group is None:
        raise ValueError('<group> is required.')
    if operation not in ('MEAN', 'SUM'):
        raise ValueError('Unsupported operation: ' + operation)
    coll_args = group.arguments.copy()
    coll_args['communication'] = 'ALLREDUCE'
    coll_args['operation'] = operation
    if context.executing_eagerly():
        return OpLib.execute('Collective', inputs, **coll_args)
    kwargs.update(coll_args)
    return OpLib.add('Collective', inputs, **kwargs)


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
        The communication group.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if group is None:
        group = distributed.get_group()
    if group is None:
        raise ValueError('<group> is required.')
    coll_args = group.arguments.copy()
    coll_args['root'] = root
    coll_args['communication'] = 'BROADCAST'
    if context.executing_eagerly():
        return OpLib.execute('Collective', inputs, **coll_args)
    kwargs.update(coll_args)
    return OpLib.add('Collective', inputs, **kwargs)
