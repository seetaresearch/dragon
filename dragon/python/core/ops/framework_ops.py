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
"""Framework ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph.op_def import OpDef
from dragon.core.eager import context
from dragon.core.ops.utils import ArgHelper
from dragon.core.ops.utils import OpSchema


def python_plugin(
    inputs,
    module_name,
    class_name,
    num_outputs=1,
    kwargs_str=None,
    no_grad=True,
    **kwargs
):
    """Create a plugin operator from the python class.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The inputs.
    module_name : str, required
        The name of module where defines the class.
    class_name : str, required
        The name of class to create.
    num_outputs : int, optional, default=1
        The number of outputs.
    kwargs_str : str, optional
        The stringify kwargs kept for class.
    no_grad : bool, optional, default=True
        **True** to truncate the gradient of inputs.

    Returns
    -------
    Sequence[dragon.Tensor]
        The outputs.

    """
    args = ArgHelper.parse(locals())
    if context.executing_eagerly():
        raise RuntimeError('Excepted the graph execution mode.')
    else:
        op_type = 'PythonPlugin' + ('Infer' if no_grad else '')
        return OpDef.apply(op_type, **args)


@OpSchema.num_inputs(1)
def stop_gradient(inputs, **kwargs):
    r"""Return the identity of input with truncated gradient-flow.

    The expression itself is unaffected, but gradient is stopped.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        An identity of input.

    """
    args = ArgHelper.parse(locals())
    if context.executing_eagerly():
        raise RuntimeError('Excepted the graph execution mode.')
    else:
        return OpDef.apply('StopGradient', **args)
