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


@OpSchema.Inputs(1)
def Cast(inputs, dtype='float32', inplace=False, **kwargs):
    """Cast the data type of inputs to a specific one.

    If ``inplace`` is ``True``, cast ``self`` instead of returning a new one.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    dtype : str
        The specific data type.
    inplace : bool
        Whether to modify the inputs.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> x = Tensor('x', dtype='float32').Variable()
    >>> y = Cast(x, 'int32')
    >>> z = x.astype('int64')
    >>> xx = x.astype('float64', inplace=True)
    >>> print(x.name, xx.name)

    """
    arguments = ParseArgs(locals())

    if inplace:
        arguments['inputs'] = []
        arguments['existing_outputs'] = [inputs]

    return Tensor.CreateOperator('Cast', **arguments)


def Run(inputs, module, op, param_str='', num_outputs=1, **kwargs):
    """Run a custom operator. (Without GradientFlow)

    **Type Constraints**: *None*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs.
    module : str
        The module.
    op : str
        The `class` under the module.
    param_str : str
        The str describing parameters.
    num_outputs : int
        The number of num_outputs.

    Returns
    -------
    sequence of Tensor
        The outputs.

    Notes
    -----
    This operator is designed to truncate gradients flow.

    If you want to flow gradients, use `Template(*args, **kwargs)`_.

    References
    ----------
    `DataProcessOp`_ - How to custom a RunOp in Dragon.

    """
    return Tensor.CreateOperator('Run', **ParseArgs(locals()))


def Template(inputs, module, op, param_str='', num_outputs=1, **kwargs):
    """Run a custom operator. (With GradientFlow)

    **Type Constraints**: *None*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs.
    module : str
        The module.
    op : str
        The `class` under the module.
    param_str : str
        The str describing parameters.
    num_outputs : int
        The number of num_outputs.

    Returns
    -------
    sequence of Tensor
        The outputs.

    References
    ----------
    `VecMultOp`_ - How to custom a TemplateOp in Dragon.

    """
    return Tensor.CreateOperator('Template', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def Accuracy(inputs, top_k=1, axis=1, ignore_labels=(), **kwargs):
    """Calculate the Top-K accuracy.

    **Type Constraints**:

    * logits (*float16*, *float32*)

    * labels (*float32*, *int64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [logits, labels].
    top_k : int
        The top-k accuracy to calculate.
    axis : int
        The axis of classes.
    ignore_labels : sequence of int
        The labels to ignore.

    Returns
    -------
    Tensor
        The top-k accuracy.

    """
    return Tensor.CreateOperator('Accuracy', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def StopGradient(inputs, **kwargs):
    """Return the identity of input with truncated gradient flow.

    The expression itself is unaffected, but gradient is stopped.

    **Type Constraints**: *None*

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A identity of input.

    """
    return Tensor.CreateOperator('StopGradient', **ParseArgs(locals()))