# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import *


def Copy(inputs, **kwargs):
    """Copy A to B.

    Parameters
    ----------
    inputs : list or Tensor
        The inputs, represent A and B respectively.

    Returns
    -------
    Tensor
        The output tensor, i.e., B(taking values of A).

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())
    arguments['existing_outputs'] = [arguments['inputs'][1]]
    arguments['inputs'] = [arguments['inputs'][0]]

    output =  Tensor.CreateOperator(nout=1, op_type='Copy', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Equal(inputs, **kwargs):
    """Equal Comparing between A and B.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent A and B respectively.

    Returns
    -------
    Tensor
        The comparing results.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='Compare',
                                   operation='EQUAL', **arguments)

    if all(input.shape is not None for input in inputs):
        output.shape = inputs[0].shape[:]

    return output