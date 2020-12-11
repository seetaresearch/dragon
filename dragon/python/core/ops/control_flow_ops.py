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
"""Control flow ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.eager import context
from dragon.core.framework import ops
from dragon.core.ops import control_flow_ops_lib
from dragon.core.ops.utils import ArgHelper
from dragon.core.ops.utils import OpSchema


@OpSchema.num_inputs(2)
@ArgHelper.repeated_desc('starts')
@ArgHelper.repeated_desc('sizes')
def assign(inputs, starts=None, sizes=None, **kwargs):
    r"""Assign the value to input.

    .. math:: \text{input}[\text{start}:\text{start} + \text{size}, ...] = \text{value}

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input and value tensor.
    starts : Sequence[Union[int, dragon.Tensor]], optional
        The start location for each dimension.
    sizes : Sequence[Union[int, dragon.Tensor]], optional
        The number of elements assigned from start.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    inputs[1] = ops.scalar_to_tensor(inputs[1], inputs[0].dtype)
    op_lib = control_flow_ops_lib.Assign
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(starts) if starts is not None else 0,
            ).apply(inputs, starts, sizes, inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(3)
def masked_assign(inputs, **kwargs):
    r"""Assign the value to input where mask is 1.

    .. math::
        \text{input}_{i} =
            \begin{cases}
                \text{value}_{i}, & \text{ if } \text{mask}_{i} = 1 \\
                \text{input}_{i}, & \text{ otherwise }
        \end{cases}

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input, value and mask tensor.

    Returns
    -------
    dragon.Tensor
        The input tensor.

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    inputs[1] = ops.scalar_to_tensor(inputs[1], inputs[0].dtype)
    op_lib = control_flow_ops_lib.MaskedAssign
    if context.executing_eagerly():
        return op_lib.instantiate().apply(inputs, inplace=inplace)
    else:
        return op_lib.blend(**args)
