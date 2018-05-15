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


def Add(inputs, **kwargs):
    """Calculate A + B.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent A and B respectively.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Add', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Sub(inputs, **kwargs):
    """Calculate A - B.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent A and B respectively.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Add', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output

def Mul(inputs, **kwargs):
    """Calculate A * B.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent A and B respectively.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output =  Tensor.CreateOperator(nout=1, op_type='Mul', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Div(inputs, **kwargs):
    """Calculate A / B.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent A and B respectively.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output =  Tensor.CreateOperator(nout=1, op_type='Div', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Clip(inputs, low=None, high=None, **kwargs):
    """Clip the input to be between lower and higher bounds.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    low : basic numerical type or None
        The lower bound. Default is ``None`` (Ignore).
    high : basic numerical type or None
        The higher bound. Default is ``None`` (Ignore).

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
    if low is not None: arguments['low'] = float(arguments['low'])
    if high is not None: arguments['high'] = float(arguments['high'])

    output = Tensor.CreateOperator(nout=1, op_type='Clip', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Matmul(inputs, TransA=False, TransB=False, **kwargs):
    """Matrix Multiplication.

    This operator can calculate a batch of matrix multiplication.

    To trigger ``Batch Matrix Multiplication``, the ``ndim`` of A must greater than ``2``.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent A and B respectively.
    TransA : boolean
        Whether to transpose A.
    TransB : boolean
        Whether to transpose B.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Matmul', **arguments)

    if inputs[0].shape is not None and \
            inputs[1].shape is not None:
        if len(inputs[0].shape) < 2 or \
            len(inputs[1].shape) < 2:
                raise ValueError('The rank of A and B should be at least 2.')
        if len(inputs[0].shape) != len(inputs[1].shape):
            raise ValueError('Both A and B should have the same number of dimensions.')
        M = inputs[0].shape[-1] if TransA else inputs[0].shape[-2]
        K1 = inputs[0].shape[-2] if TransA else inputs[0].shape[-1]
        K2 = inputs[1].shape[-1] if TransB else inputs[1].shape[-2]
        N = inputs[1].shape[-2] if TransB else inputs[1].shape[-1]
        if K1 != K2:
            raise ValueError('Can not multiply A: ({}, {}} with B: ({}, {})'.format(M, K1, K2, N))
        output.shape = inputs[0].shape[:]
        output.shape[-2] = M
        output.shape[-1] = N

    return output


def Dot(inputs, TransA=False, TransB=False, **kwargs):
    """DotProduct Function.

    This operator can trigger ``Matrix Multiplication`` or ``Matrix Vector Multiplication`` also.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent A and B respectively.
    TransA : boolean
        Whether to transpose A.
    TransB : boolean
        Whether to transpose B.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Dot', **arguments)

    if inputs[0].shape is not None and inputs[1].shape is not None:
        a_shape = inputs[0].shape[:] if not TransA else inputs[0].shape[::-1]
        b_shape = inputs[1].shape[:] if not TransB else inputs[1].shape[::-1]
        output.shape = a_shape
        output.shape[-1] = b_shape[-1]

    return output


def InnerProduct(inputs, num_output, axis=1, TransW=True, **kwargs):
    """InnerProduct Function.

    The number of inputs vary from ``2`` to ``3`` (Without or With ``bias``).

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, weights, bias].
    num_output : int
        The output dim.
    axis : int
        The start axis to calculate, can be negative.
    TransW : boolean
        Whether to transpose the weights.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 2, 3)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='InnerProduct', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[: axis + 1]
        output.shape[axis] = num_output

    return output


def Eltwise(inputs, operation='SUM', coeffs=None, **kwargs):
    """Eltwise Sum/Product Function.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs.
    operation : str
        The operation, ``SUM`` or ``PROD``.
    coeffs : list of float or None
        The coefficients on inputs. Default is ``None`` (All are ``1.0``) .

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 2, INT_MAX)
    arguments = ParseArguments(locals())
    if arguments['coeffs'] is not None:
        arguments['coeffs'] = [float(ele) for ele in arguments['coeffs']]

    output = Tensor.CreateOperator(nout=1, op_type='Eltwise', **arguments)

    if all(input.shape is not None for input in inputs):
        output.shape = inputs[0].shape[:]

    return output


def Log(inputs, **kwargs):
    """Calculate the logarithm of input.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The logarithm tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Log', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Exp(inputs, **kwargs):
    """Calculate the exponential of input.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The exponential result.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Exp', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Pow(inputs, power, shift=None, scale=None, **kwargs):
    """Calculate the power of input.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    power : float
        The power factor.
    shift : float or None
        The shift magnitude. Default is ``None`` (Ignore).
    scale : float or None
        The scale factor. Default is ``None`` (Ignore).

    Returns
    -------
    Tensor
        The power result, calculated as: |power_function|

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    arguments['power']= float(power)
    if arguments['scale'] is not None: arguments['scale'] = float(scale)
    if arguments['shift'] is not None: arguments['shift'] = float(shift)

    output =  Tensor.CreateOperator(nout=1, op_type='Pow', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Square(inputs, **kwargs):
    """Calculate the square of input.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The square result.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Square', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Sqrt(inputs, **kwargs):
    """Calculate the sqrt of input.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The sqrt result.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())
    arguments['power'] = 0.5

    output = Tensor.CreateOperator(nout=1, op_type='Pow', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Scale(inputs, axis=1, num_axes=1, **kwargs):
    """Scale Function.

    The number of inputs vary from ``2`` to ``3`` (Without or With ``bias``).

    The scale ranges are: |scale_function|

    Set ``axis`` to specific the start axis(can be negative).

    Set ``num_axes`` to -1 will scale all remained axes.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, scale, bias].
    axis : int
        The start axis to scale.
    num_axes : int
        The number of axes to scale.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 2, 3)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Scale', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def GramMatrix(inputs, axis=1, **kwargs):
    """Calculate the gram matrix, introduced by `[Gatys et.al, 2016] <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf>`_.

    Parameters
    ---------=
    inputs : Tensor
        The input tensor.
    axis : int
        The start axis to calculate.

    Returns
    -------
    Tensor
        The output tensor, calculated as: |gram_matrix_function|

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='GramMatrix', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[: axis + 2]
        output.shape[axis + 1] = output.shape[axis]

    return output