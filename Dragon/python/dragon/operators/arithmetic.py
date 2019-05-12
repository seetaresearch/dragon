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


@OpSchema.Inputs(2)
def Add(inputs, **kwargs):
    """Calculate *A + B*.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, A and B.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Add', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def Sub(inputs, **kwargs):
    """Calculate *A - B*.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, A and B.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Sub', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def Mul(inputs, **kwargs):
    """Calculate *A * B*.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, A and B.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Mul', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def Div(inputs, **kwargs):
    """Calculate *A / B*.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, A and B.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Div', **ParseArgs(locals()))


@OpSchema.ConvertConstantInputs()
@OpSchema.Inputs(2)
def Maximum(inputs, **kwargs):
    """Return the max value of given two inputs.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence
        The inputs, can be either Tensor or number.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Maximum', **ParseArgs(locals()))


@OpSchema.ConvertConstantInputs()
@OpSchema.Inputs(2)
def Minimum(inputs, **kwargs):
    """Return the min value of given two inputs.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence
        The inputs, can be either Tensor or number.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Minimum', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Moments(inputs, axes=None, keep_dims=False, **kwargs):
    """Calculate the mean and variance of inputs along the given axes.

    The data type of moments will be *float32* typically,
    except the *float64* inputs (*float64* moments instead).

    If ``axes`` is *None*, a Scalar will be returned.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    axes : int or sequence of int, optional
        The axes to compute the moments.
    keep_dims : bool, optional
        Whether to keep the reduced dimensions of moments.

    Returns
    -------
    sequence of Tensor
        The mean and variance.

    """
    arguments = ParseArgs(locals())
    if axes and not isinstance(axes, (tuple, list)): arguments['axes'] = [axes]
    return Tensor.CreateOperator('Moments', num_outputs=2, **arguments)


@OpSchema.Inputs(1)
def Clip(inputs, low=None, high=None, **kwargs):
    """Clip the input to be between lower and higher bounds.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    low : number, optional
        The lower bound.
    high : number, optional
        The higher bound.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())
    if low is not None: arguments['low'] = float(arguments['low'])
    if high is not None: arguments['high'] = float(arguments['high'])
    return Tensor.CreateOperator(op_type='Clip', **arguments)


@OpSchema.Inputs(2)
def Matmul(inputs, transA=False, transB=False, **kwargs):
    """Matrix Multiplication.

    This operator can calculate a batch of matrix multiplication.

    **Type Constraints**: (*float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, A and B.
    transA : bool, optional, default=False
        Whether to transpose A.
    transB : bool, optional, default=False
        Whether to transpose B.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Matmul', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def Dot(inputs, transA=False, transB=False, **kwargs):
    """Calculate the Vector dot.

    This operator can trigger *Matrix Multiplication (Right Alignment)* or

    *Matrix Vector Multiplication (Right Alignment)* also.

    **Type Constraints**: (*float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, A and B.
    transA : bool, optional, default=False
        Whether to transpose A.
    transB : bool, optional, default=False
        Whether to transpose B.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Dot', **ParseArgs(locals()))


@OpSchema.Inputs(2, 3)
def FullyConnected(inputs, num_output, axis=1, transW=True, **kwargs):
    """Calculate *Y = X * W' + b*.

    Where *W'* = *Transpose(W)* if ``transW`` else *W*.

    **Type Constraints**: (*float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [X, W] + [b].
    num_output : int
        The output dim.
    axis : int, optional, default=1
        The start axis to calculate, can be negative.
    transW : bool, optional, default=True
        Whether to transpose the W.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('FullyConnected', **ParseArgs(locals()))


@OpSchema.Inputs(2, INT_MAX)
def Eltwise(inputs, operation='SUM', coef=None, **kwargs):
    """Element-wise Sum or Product the arbitrary number of inputs.

    If ``coef`` is *None*, *1.0* will be used instead.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs.
    operation : {*SUM*, *PROD*}, optional
        The operation to apply.
    coef : sequence of number, optional
        The coefficients on inputs.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())
    if arguments['coef'] is not None:
        arguments['coef'] = [float(e) for e in arguments['coef']]
    return Tensor.CreateOperator('Eltwise', **arguments)


@OpSchema.Inputs(1)
def Log(inputs, **kwargs):
    """Calculate the logarithm of input.

    **Type Constraints**: (*float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The logarithm tensor.

    """
    return Tensor.CreateOperator('Log', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Exp(inputs, **kwargs):
    """Calculate the exponential of input.

    **Type Constraints**: (*float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The exponential result.

    """
    return Tensor.CreateOperator('Exp', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Pow(inputs, power, shift=0., scale=1., **kwargs):
    """Calculate the power of input.

    Formulation: |power_function|

    **Type Constraints**: (*float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    power : float, required
        The power factor.
    shift : float, optional, default=0.
        The shift magnitude.
    scale : float, optional, default=1.
        The scale factor.

    Returns
    -------
    Tensor
        The powered result.

    """
    arguments = ParseArgs(locals())
    arguments['power']= float(power)
    if arguments['scale'] is not None: arguments['scale'] = float(scale)
    if arguments['shift'] is not None: arguments['shift'] = float(shift)
    return Tensor.CreateOperator('Pow', **arguments)


@OpSchema.Inputs(1)
def Square(inputs, **kwargs):
    """Calculate the square of input.

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The square result.

    """
    return Tensor.CreateOperator('Square', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def Sqrt(inputs, **kwargs):
    """Calculate the sqrt of input.

    **Type Constraints**: (*float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The sqrt result.

    """
    return Tensor.CreateOperator('Sqrt', **ParseArgs(locals()))


@OpSchema.Inputs(2, 3)
def Affine(inputs, axis=1, num_axes=1, **kwargs):
    """Calculate *Y = Ax + b* along the given range of axes.

    The scale ranges are: |scale_function|

    Set ``axis`` to specific the start axis.

    Set ``num_axes`` to -1 will scale all remained axes.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [x, A] + [b].
    axis : int, optional, default=1
        The start axis to scale, can be negative.
    num_axes : int, optional, default=1
        The number of axes to scale.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('Affine', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def GramMatrix(inputs, axis=1, **kwargs):
    """Calculate the gram matrix. `[Gatys et.al, 2016] <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf>`_.

    Formulation: |gram_matrix_function|

    **Type Constraints**: *float32*

    Parameters
    ---------=
    inputs : Tensor
        The input tensor.
    axis : int, optional, default=1
        The start axis to calculate.

    Returns
    -------
    Tensor
        The gram matrix.

    """
    return Tensor.CreateOperator('GramMatrix', **ParseArgs(locals()))


@OpSchema.Inputs(1, INT_MAX)
def Accumulate(inputs, alpha=1., beta=1., **kwargs):
    """Calculate *y = alpha * x + beta * y*

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, i.e., the *x*.
    alpha : float, optional, default=1.
        The value of alpha.
    beta : float, optional, default=1.
        The value beta.

    Returns
    -------
    sequence of Tensor
        The outputs, i.e., the *y*.

    """
    return Tensor.CreateOperator('Accumulate', **ParseArgs(locals()))


@OpSchema.Inputs(1, INT_MAX)
def MovingAverage(inputs, decay, **kwargs):
    """Calculate the *y = (1 - decay) * x + decay * y*

    **Type Constraints**: (*int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, i.e., the *x*.
    decay : float, required
        The decay factor.

    Returns
    -------
    sequence of Tensor
        The outputs, i.e., the *y*.

    """
    return Accumulate(inputs, 1. - decay, decay, **kwargs)