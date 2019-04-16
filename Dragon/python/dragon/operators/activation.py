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
def Relu(inputs, **kwargs):
    """Rectified Linear Unit function. `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |relu_function|.

    """
    return Tensor.CreateOperator('Relu', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def LRelu(inputs, slope=0.2, **kwargs):
    """Leaky Rectified Linear Unit function.

    **Type Constraints**: (*float16*, *float32*)

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
    return Tensor.CreateOperator('Relu', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def PRelu(inputs, channel_shared=False, data_format='NCHW', **kwargs):
    """Parametric Rectified Linear Unit function. `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The input and trainable parameter(slope).
    channel_shared : bool
        Whether to share the parameter(slope) across channels.
    data_format : str
        The data format, ``NCHW`` or ``NHWC``.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |prelu_function|

    """
    return Tensor.CreateOperator('PRelu', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Elu(inputs, alpha=1.0, **kwargs):
    """Exponential Linear Unit function. `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    **Type Constraints**: (*float16*, *float32*)

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
    return Tensor.CreateOperator('Elu', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def SElu(inputs, **kwargs):
    """Scaled Exponential Linear Unit function. `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |selu_function|

    """
    return Tensor.CreateOperator('SElu', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Sigmoid(inputs, **kwargs):
    """Sigmoid function.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |sigmoid_function|.

    """
    return Tensor.CreateOperator('Sigmoid', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Tanh(inputs, **kwargs):
    """Tanh function.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |tanh_function|.

    """
    return Tensor.CreateOperator('Tanh', **ParseArgs(locals()))


@OpSchema.Inputs(1)
@ArgumentHelper.Desc('prob', as_target=True)
def Dropout(inputs, prob=0.5, scale=True, **kwargs):
    """Randomly set a unit into zero. `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    prob : float or Tensor
        The prob of dropping. Default is ``0.5``.
    scale : bool
        Whether to scale the output during training.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |dropout_function|.

    """
    return Tensor.CreateOperator('Dropout', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Softmax(inputs, axis=1, **kwargs):
    """Softmax function.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axis : int
        The axis to apply softmax, can be negative.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |softmax_function|.

    """
    return Tensor.CreateOperator('Softmax', **ParseArgs(locals()))