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
    """Copy A to B.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, A and B respectively.

    Returns
    -------
    Tensor
        The output tensor, i.e., B(taking values of A).

    """
    arguments = ParseArgs(locals())
    arguments['existing_outputs'] = [arguments['inputs'][1]]
    arguments['inputs'] = [arguments['inputs'][0]]
    return Tensor.CreateOperator('Copy', **arguments)


@OpSchema.Inputs(2)
def Equal(inputs, to_uint8=False, **kwargs):
    """``Equal`` comparing between A and B.

    Set ``to_uint8`` if you expect the ``uint8`` results instead of ``bool``.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent A and B respectively.
    to_uint8 : bool
        ``True`` to convert to ``uint8`` results.

    Returns
    -------
    Tensor
        The comparing results.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Compare', operation='EQUAL', **arguments)


@OpSchema.Inputs(2)
def Less(inputs, to_uint8=False, **kwargs):
    """``Less`` comparing between A and B.

    Set ``to_uint8`` if you expect the ``uint8`` results instead of ``bool``.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent A and B respectively.
    to_uint8 : bool
        ``True`` to convert to ``uint8`` results.

    Returns
    -------
    Tensor
        The comparing results.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Compare', operation='LESS', **arguments)


@OpSchema.Inputs(2)
def Greater(inputs, to_uint8=False, **kwargs):
    """``Less`` comparing between A and B.

    Set ``to_uint8`` if you expect the ``uint8`` results instead of ``bool``.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent A and B respectively.
    to_uint8 : bool
        ``True`` to convert to ``uint8`` results.

    Returns
    -------
    Tensor
        The comparing results.

    """
    arguments = ParseArgs(locals())
    return Tensor.CreateOperator('Compare', operation='GREATER', **arguments)