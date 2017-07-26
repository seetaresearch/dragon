# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor

def Add(inputs, **kwargs):
    """

    :param inputs:      a list of 2 Tensors
    :return:            a Tensor of input[0] + input[1]

    """
    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('Add Operator accepts a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Add', **kwargs)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Sub(inputs, **kwargs):
    """

    :param inputs:      a list of 2 Tensors
    :return:            a Tensor of input[0] - input[1]

    """
    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('Sub Operator accepts a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Sub', **kwargs)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Mul(inputs, **kwargs):
    """

    :param inputs:      a list of 2 Tensors
    :return:            a Tensor of input[0] * input[1]

    """
    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('Mul Operator accepts a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Mul', **kwargs)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Div(inputs, **kwargs):
    """

    :param inputs:      a list of 2 Tensors
    :return:            a Tensor of input[0] / input[1]

    """
    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('Div Operator accepts a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Div', **kwargs)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Clip(inputs, low=None, high=None, **kwargs):
    """
    :param inputs:       a Tensor with any shape
    :return:             a Tensor of clip(x)

        """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Log Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    if low is None: del kwargs['low']
    if high is None: del kwargs['high']

    output = Tensor.CreateOperator(nout=1, op_type='Clip', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Matmul(inputs, TransA=False, TransB=False, **kwargs):
    """

    :param inputs:      a list of 2 Tensors with same shape
    :return:            a Tensor of input[0] * input[1]

    """
    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('Matmul Operator accepts a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Matmul', **kwargs)

    if inputs[0].shape is not None and inputs[1].shape is not None:
        pass

    return output


def Dot(inputs, TransA=False, TransB=False, **kwargs):
    """

    :param inputs:      a list of 2 Tensors
    :return:            a Tensor of input[0] \dot input[1]

    """
    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('Dot Operator accepts a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Dot', **kwargs)

    if inputs[0].shape is not None and inputs[1].shape is not None:
        a_shape = inputs[0].shape[:] if not TransA else inputs[0].shape[::-1]
        b_shape = inputs[1].shape[:] if not TransB else inputs[1].shape[::-1]
        output.shape = a_shape
        output.shape[-1] = b_shape[-1]

    return output


def InnerProduct(inputs, num_output, axis=1, TransW=True, **kwargs):
    """

    :param inputs:       a list contains [input, weight] or [input, weight, bias]
    :param num_output:   a int of the output dim
    :param axis          a int of the start axis
    :param TransW        a bool of whether to transpose the weights
    :return:             a Tensor of { input * weight + bias }

    """
    if not isinstance(inputs, list) or len(inputs) < 2:
        raise RuntimeError('InnerProduct Operator accpets a list of at least 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='InnerProduct', **kwargs)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[: axis + 1]
        output.shape[axis] = num_output

    return output


def Eltwise(inputs, operation='SUM', coeffs=None, **kwargs):
    """

    :param inputs:       a list several Tensors with same shape
    :param operation:    a str of 'SUM' or 'PRODUCT'
    :param coeffs:       a float list of coeffs (None uses 1.0)
    :return:             a Tensor of Operation{ input(0), input(1), ... }

    """
    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    if kwargs['coeffs'] is None: del kwargs['coeffs']

    output = Tensor.CreateOperator(nout=1, op_type='Eltwise', **kwargs)

    if all(input.shape is not None for input in inputs):
        output.shape = inputs[0].shape[:]

    return output


def Log(inputs, **kwargs):
    """
    :param inputs:       a Tensor with any shape
    :return:             a Tensor of log(x)

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Log Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Log', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Exp(inputs, **kwargs):
    """
    :param inputs:       a Tensor with any shape
    :return:             a Tensor of exp(x)

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Exp Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Exp', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Pow(inputs, power, shift=None, scale=None, **kwargs):
    """

    :param inputs:       a Tensor with any shape
    :param power:        a float of power
    :param shift:        a float of shift
    :param scale:        a float of scale
    :return:             a Tensor of { [(x + shift) * scale] ^ power }

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Pow Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    kwargs['power']= float(power)
    if kwargs['scale'] is not None: kwargs['scale'] = float(scale)
    if kwargs['shift'] is not None: kwargs['shift'] = float(shift)

    output =  Tensor.CreateOperator(nout=1, op_type='Pow', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Square(inputs, **kwargs):
    """

    :param inputs:       a Tensor with any shape
    :return:             a Tensor of x^{2}

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Square Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Square', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def Scale(inputs, axis=1, num_axes=1, **kwargs):

    if not isinstance(inputs, list) or len(inputs) < 2:
        raise RuntimeError('Scale Operator accpets a list of at least 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Scale', **kwargs)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Argmax(inputs, axis=0, top_k=1, **kwargs):

    if not isinstance(inputs, Tensor):
        raise RuntimeError('Argmax Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Argmax', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        if top_k > 1: output.shape[axis] = top_k
        else: del output.shape[axis]

    return output


def GramMatrix(inputs, axis=1, **kwargs):

    if not isinstance(inputs, Tensor):
        raise RuntimeError('GramMatrix Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='GramMatrix', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[: axis + 2]
        output.shape[axis + 1] = output.shape[axis]

    return output