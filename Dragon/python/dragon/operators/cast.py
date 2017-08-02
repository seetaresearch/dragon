# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor

def FloatToHalf(inputs, **kwargs):
    """

    :param inputs:      a Tensor with type of float32
    :return:            a Tensor with type of float16

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Relu Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='FloatToHalf', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output