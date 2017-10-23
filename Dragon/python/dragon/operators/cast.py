# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from . import *

def FloatToHalf(inputs, **kwargs):
    """Cast the type of tensor from ``float32`` to ``float16``.

    Parameters
    ----------
    inputs : Tensor
        The ``float32`` tensor.

    Returns
    -------
    Tensor
        The ``float16`` tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='FloatToHalf', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output