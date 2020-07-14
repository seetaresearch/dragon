# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""The control flow ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.eager import context
from dragon.core.framework import ops
from dragon.core.ops import control_flow_ops_lib
from dragon.core.ops.utils import ArgHelper
from dragon.core.ops.utils import OpSchema
from dragon.core.ops.utils import parse_args
from dragon.core.util import nest


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
        The input tensor.

    """
    args = parse_args(locals())
    inputs[1] = ops.scalar_to_tensor(inputs[1], inputs[0].dtype)
    op_lib = control_flow_ops_lib.Assign
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(starts) if starts is not None else 0,
            ).apply(inputs, starts, sizes)
    else:
        args['outputs'] = [args['inputs'][0]]
        args['inputs'] = [args['inputs'][1]]
        return op_lib.blend(**args)


@OpSchema.num_inputs(1, 2)
def copy(inputs, **kwargs):
    """Copy the input.

    Examples:

    ```python
    # Copy ``x`` to ``y``
    x = dragon.ones(shape=(2, 3))
    y = dragon.zeros(shape=(2, 4))
    dragon.copy([x, y])

    # Copy to a new tensor from ``x``
    y = dragon.copy(x)
    ```

    Parameters
    ----------
    inputs : Union[dragon.Tensor, Sequence[dragon.Tensor]]
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    args['inputs'] = nest.flatten(inputs)
    if len(args['inputs']) == 2:
        args['outputs'] = [args['inputs'][1]]
        args['inputs'] = [args['inputs'][0]]
    else:
        args['outputs'] = None
    op_lib = control_flow_ops_lib.Copy
    if context.executing_eagerly():
        return op_lib \
            .instantiate() \
            .apply(args['inputs'], args['outputs'])
    else:
        return op_lib.blend('Copy', **args)


@OpSchema.num_inputs(3)
def masked_assign(inputs, **kwargs):
    r"""Assign the value to input where mask is 1.

    .. math::
        \text{input}[i] =
            \begin{cases}
                \text{value}[i], & \text{ if } \text{mask}[i] = 1 \\
                \text{input}[i], & \text{ otherwise }
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
    args = parse_args(locals())
    inputs[1] = ops.scalar_to_tensor(inputs[1], inputs[0].dtype)
    op_lib = control_flow_ops_lib.MaskedAssign
    if context.executing_eagerly():
        return op_lib.instantiate().apply(inputs)
    else:
        args.update({
            'outputs': [args['inputs'][0]],
            'inputs': [args['inputs'][1:]],
        })
        return op_lib.blend(**args)
