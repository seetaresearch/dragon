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
def Copy(inputs, **kwargs):
    """Copy the ``value`` to ``ref``.

    The size of ``value`` and ``ref`` should be same.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The ``ref`` and ``value`` respectively.

    Returns
    -------
    Tensor
        The ``ref``.

    """
    arguments = ParseArgs(locals())
    arguments['existing_outputs'] = [arguments['inputs'][0]]
    arguments['inputs'] = [arguments['inputs'][1]]
    return Tensor.CreateOperator('Copy', **arguments)


@OpSchema.ConvertConstantInputs()
@OpSchema.Inputs(2)
@ArgumentHelper.RepeatedDesc('starts')
@ArgumentHelper.RepeatedDesc('sizes')
def Assign(inputs, starts=None, sizes=None, **kwargs):
    """Assign the ``value`` to ``ref``.

    The value of ``sizes`` could be set to *-1* (to end) or *0* (squeeze).

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The ``ref`` and ``value`` respectively.
    starts : sequence of (int, Tensor), optional
        The start pos of each dimension.
    sizes : sequence of (int, Tensor), optional
        The size of each dimension.

    Returns
    -------
    Tensor
        The ``ref``.

    """
    arguments = ParseArgs(locals())
    arguments['existing_outputs'] = [arguments['inputs'][0]]
    arguments['inputs'] = [arguments['inputs'][1]]
    return Tensor.CreateOperator('Assign', **arguments)


@OpSchema.ConvertConstantInputs()
@OpSchema.Inputs(2)
def MaskedAssign(inputs, mask, **kwargs):
    """Assign the ``value`` to ``ref`` where ``mask`` is *1*.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The ``ref`` and ``value`` respectively.
    mask : Tensor
        The mask, with the same size as ``ref``.

    Returns
    -------
    Tensor
        The ``ref``.

    """
    arguments = ParseArgs(locals())
    arguments['existing_outputs'] = [arguments['inputs'][0]]
    arguments['inputs'] = [arguments['inputs'][1], mask]
    return Tensor.CreateOperator('Assign', **arguments)


@OpSchema.ConvertConstantInputs()
@OpSchema.Inputs(2)
def Equal(inputs, to_uint8=False, **kwargs):
    """*Equal* comparing between A and B.

    Set ``to_uint8`` if you expect the *uint8* results instead of *bool*.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent A and B respectively.
    to_uint8 : bool
        *True* to convert to *uint8* results.

    Returns
    -------
    Tensor
        The comparing results.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Compare', operation='EQ', **arguments)


@OpSchema.Inputs(2)
def NotEqual(inputs, to_uint8=False, **kwargs):
    """*NotEqual* comparing between A and B.

    Set ``to_uint8`` if you expect the *uint8* results instead of *bool*.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent A and B respectively.
    to_uint8 : bool
        *True* to convert to *uint8* results.

    Returns
    -------
    Tensor
        The comparing results.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Compare', operation='NE', **arguments)



@OpSchema.ConvertConstantInputs()
@OpSchema.Inputs(2)
def Less(inputs, to_uint8=False, **kwargs):
    """*Less* comparing between A and B.

    Set ``to_uint8`` if you expect the *uint8* results instead of *bool*.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent A and B respectively.
    to_uint8 : bool
        *True* to convert to *uint8* results.

    Returns
    -------
    Tensor
        The comparing results.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Compare', operation='LT', **arguments)


@OpSchema.ConvertConstantInputs()
@OpSchema.Inputs(2)
def LessEqual(inputs, to_uint8=False, **kwargs):
    """*LessEqual* comparing between A and B.

    Set ``to_uint8`` if you expect the *uint8* results instead of *bool*.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent A and B respectively.
    to_uint8 : bool
        *True* to convert to *uint8* results.

    Returns
    -------
    Tensor
        The comparing results.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Compare', operation='LE', **arguments)


@OpSchema.ConvertConstantInputs()
@OpSchema.Inputs(2)
def Greater(inputs, to_uint8=False, **kwargs):
    """*Greater* comparing between A and B.

    Set ``to_uint8`` if you expect the *uint8* results instead of *bool*.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent A and B respectively.
    to_uint8 : bool
        *True* to convert to *uint8* results.

    Returns
    -------
    Tensor
        The comparing results.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Compare', operation='GT', **arguments)


@OpSchema.ConvertConstantInputs()
@OpSchema.Inputs(2)
def GreaterEqual(inputs, to_uint8=False, **kwargs):
    """*GreaterEqual* comparing between A and B.

    Set ``to_uint8`` if you expect the *uint8* results instead of *bool*.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent A and B respectively.
    to_uint8 : bool
        *True* to convert to *uint8* results.

    Returns
    -------
    Tensor
        The comparing results.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Compare', operation='GE', **arguments)