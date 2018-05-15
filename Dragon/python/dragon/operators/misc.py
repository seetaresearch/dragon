# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import *


def Run(inputs, module, op, param_str='', nout=1, **kwargs):
    """Run a custom operator. (Without GradientFlow)

    Parameters
    ----------
    inputs : list of Tensor
        The inputs.
    module : str
        The module.
    op : str
        The `class` under the module.
    param_str : str
        The str describing parameters.
    nout : int
        The number of output.

    Returns
    -------
    Tensor or list of Tensor
        The outputs.

    Notes
    -----
    This operator is designed to truncate gradients flow.

    If you want to flow gradients, use `Template(*args, **kwargs)`_.

    References
    ----------
    `DataProcessOp`_ - How to custom a RunOp in Dragon.

    """
    arguments = ParseArguments(locals())
    return Tensor.CreateOperator(op_type='Run', **arguments)


def Template(inputs, module, op, param_str='', nout=1, **kwargs):
    """Run a custom operator. (With GradientFlow)

    Parameters
    ----------
    inputs : list of Tensor
        The inputs.
    module : str
        The module.
    op : str
        The `class` under the module.
    param_str : str
        The str describing parameters.
    nout : int
        The number of output.

    Returns
    -------
    Tensor or list of Tensor
        The outputs.

    References
    ----------
    `VecMultOp`_ - How to custom a TemplateOp in Dragon.

    """
    arguments = ParseArguments(locals())
    return Tensor.CreateOperator(op_type='Template', **arguments)


def Accuracy(inputs, top_k=1, axis=1, ignore_labels=[], **kwargs):
    """Calculate the Top-K accuracy.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, labels].
    top_k : int
        The top-k accuracy to calculate.
    axis : int
        The axis of classes.
    ignore_labels : list of int
        The labels to ignore.

    Returns
    -------
    Tensor
        The top-k accuracy.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output =  Tensor.CreateOperator(nout=1, op_type='Accuracy', **arguments)
    output.shape = [1]
    return output


def StopGradient(inputs, **kwargs):
    """Return the identity of input with truncated gradient flow.

    The expression itself is unaffected, but gradient is stopped.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The identity of input.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output =  Tensor.CreateOperator(nout=1, op_type='StopGradient', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def MovingAverage(inputs, decay, **kwargs):
    """Calculate the moving average.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [variable, value].
    decay : float
        The decay factor.

    Returns
    -------
    Tensor
        The output tensor, i.e., ``variable``, calculated as:

        |moving_average_function|

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())
    variable = arguments['inputs'][0]
    del arguments['inputs'][0]

    output = Tensor.CreateOperator(op_type='MovingAverage',
                                   existing_outputs=variable, **arguments)
    return output