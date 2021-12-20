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

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema


def python_plugin(
    inputs,
    module_name,
    class_name,
    num_outputs=1,
    kwargs_str=None,
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
        The stringify keyword arguments.

    Returns
    -------
    Sequence[dragon.Tensor]
        The outputs.

    """
    if context.executing_eagerly():
        raise RuntimeError('Excepted the graph execution mode.')
    return OpLib.add('PythonPlugin', inputs, module_name=module_name,
                     class_name=class_name, num_outputs=num_outputs,
                     kwargs_str=kwargs_str, **kwargs)


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
    if context.executing_eagerly():
        raise RuntimeError('Excepted the graph execution mode.')
    return OpLib.add('GradientStop', inputs, **kwargs)
