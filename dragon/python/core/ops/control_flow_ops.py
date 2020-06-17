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
    r"""Assign the value to ref.

    .. math:: \text{Ref}[start:start + size, ...] = \text{Value}

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The **ref** and **value**.
    starts : Sequence[Union[int, dragon.Tensor]], optional
        The start pos of each dimension.
    sizes : Sequence[Union[int, dragon.Tensor]], optional
        The size of each dimension.

    Returns
    -------
    dragon.Tensor
        The **ref**.

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
    r"""Copy the value to ref.

    .. math:: \text{Ref}[:] = \text{Value}[:]

    Examples:

    ```python
    # Copy the content from ``x`` to ``xx``
    x = dragon.ones(shape=(2, 3))
    xx = dragon.zeros(shape=(2, 4))
    dragon.copy([xx, x])

    # Create a new tensor initialized from ``x``
    xxx = dragon.copy(x)
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The **ref** and **value**.

    Returns
    -------
    dragon.Tensor
        The **ref**.

    """
    args = parse_args(locals())
    inputs = nest.flatten(inputs)
    if len(inputs) == 2:
        args['inputs'] = [inputs[1]]
        args['outputs'] = [inputs[0]]
    else:
        args['outputs'] = None
    op_lib = control_flow_ops_lib.Copy
    if context.executing_eagerly():
        return op_lib \
            .instantiate() \
            .apply(args['inputs'], args['outputs'])
    else:
        return op_lib.blend('Copy', **args)


@OpSchema.num_inputs(2)
def masked_assign(inputs, mask, **kwargs):
    r"""Assign the value to ref where mask is **1**.

    .. math::
        \text{Ref}[i] =
        \begin{cases}
            \text{Value}[i], & \text{ if } \text{Mask}[i] = 1 \\
            \text{Ref}[i], & \text{ otherwise }
        \end{cases}

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The **ref** and **value**.
    mask : dragon.Tensor
        The mask, with the same size as **ref**.

    Returns
    -------
    dragon.Tensor
        The **ref**.

    """
    args = parse_args(locals())
    inputs[1] = ops.scalar_to_tensor(inputs[1], inputs[0].dtype)
    op_lib = control_flow_ops_lib.MaskedAssign
    if context.executing_eagerly():
        return op_lib.instantiate().apply(inputs, mask)
    else:
        args.update({
            'outputs': [args['inputs'][0]],
            'inputs': [args['inputs'][1], mask],
        })
        return op_lib.blend(**args)
