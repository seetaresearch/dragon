# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
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


def Relu(inputs, **kwargs):
    """Rectified Linear Unit function, introduces by `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |relu_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Relu', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def LRelu(inputs, slope=0.2, **kwargs):
    """Leaky Rectified Linear Unit function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    slope : float
        The slope of negative side.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |lrelu_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Relu', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def PRelu(inputs, channel_shared=False, data_format='NCHW', **kwargs):
    """Parametric Rectified Linear Unit function, introduces by `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

    Parameters
    ----------
    inputs : list of Tensor
        The input and trainable parameter(slope).
    channel_shared : boolean
        Whether to share the parameter(slope) across channels.
    data_format : str
        The data format, ``NCHW`` or ``NHWC``.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |prelu_function|

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='PRelu', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Elu(inputs, alpha=1.0, **kwargs):
    """Exponential Linear Unit function, introduces by `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    alpha : float
        The alpha.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |elu_function|

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Elu', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def SElu(inputs, **kwargs):
    """Scaled Exponential Linear Unit function, introduces by `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |selu_function|

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='SElu', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Sigmoid(inputs, **kwargs):
    """Sigmoid function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |sigmoid_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Sigmoid', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Tanh(inputs, **kwargs):
    """Tanh function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |tanh_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Tanh', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Dropout(inputs, prob=0.5, scale=True, **kwargs):
    """Randomly set a unit into zero, introduced by `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    prob : float or Tensor
        The prob of dropping. Default is ``0.5``.
    scale : boolean
        Whether to scale the output during training.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |dropout_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
    arguments = AddArgumentWithDesc(arguments, prob, 'prob', as_target=False)

    output = Tensor.CreateOperator(nout=1, op_type='Dropout', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Softmax(inputs, axis=1, **kwargs):
    """Softmax function.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The axis to perform softmax.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |softmax_function|.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Softmax', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output