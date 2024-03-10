# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Distributed operators."""

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema
from dragon.core.distributed import backend as dist_backend


@OpSchema.num_inputs(1)
def all_gather(inputs, group=None, **kwargs):
    """Gather input across all nodes.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if group is None:
        group = dist_backend.get_group()
    if group is None:
        raise ValueError("<group> is required.")
    coll_args = group.arguments.copy()
    coll_args["operation"] = "ALLGATHER"
    if context.executing_eagerly():
        return OpLib.execute("Collective", inputs, **coll_args)
    kwargs.update(coll_args)
    return OpLib.add("Collective", inputs, **kwargs)


@OpSchema.num_inputs(1)
def all_reduce(inputs, reduction="sum", group=None, **kwargs):
    """Reduce input across all nodes.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    reduction : str, optional, default='sum'
        The reduction method.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if group is None:
        group = dist_backend.get_group()
    if group is None:
        raise ValueError("<group> is required.")
    coll_args = group.arguments.copy()
    coll_args["operation"] = "ALLREDUCE"
    coll_args["reduction"] = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute("Collective", inputs, **coll_args)
    kwargs.update(coll_args)
    return OpLib.add("Collective", inputs, **kwargs)


@OpSchema.num_inputs(1)
def broadcast(inputs, root=0, group=None, **kwargs):
    """Broadcast input from the root node.

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
        group = dist_backend.get_group()
    if group is None:
        raise ValueError("<group> is required.")
    coll_args = group.arguments.copy()
    coll_args["root"] = root
    coll_args["operation"] = "BROADCAST"
    if context.executing_eagerly():
        return OpLib.execute("Collective", inputs, **coll_args)
    kwargs.update(coll_args)
    return OpLib.add("Collective", inputs, **kwargs)


@OpSchema.num_inputs(1)
def reduce_scatter(inputs, reduction="sum", group=None, **kwargs):
    """Reduce and scatter input across all nodes.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    reduction : str, optional, default='sum'
        The reduction method.
    group : ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if group is None:
        group = dist_backend.get_group()
    if group is None:
        raise ValueError("<group> is required.")
    coll_args = group.arguments.copy()
    coll_args["operation"] = "REDUCESCATTER"
    coll_args["reduction"] = reduction.upper()
    if context.executing_eagerly():
        return OpLib.execute("Collective", inputs, **coll_args)
    kwargs.update(coll_args)
    return OpLib.add("Collective", inputs, **kwargs)
