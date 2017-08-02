# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor

def Run(inputs, module, op, param_str='', nout=1, **kwargs):
    """
    :param inputs:        a list of Tensor contains inputs
    :param module:        a str of the python module
    :param op:            a str of the operator class
    :param param_str:     a str of param_str to be used in operator class
    :param nout:          a int of returned tensors
    :return:
    """

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(op_type='Run', **kwargs)


def Template(inputs, module, op, param_str='', nout=1, **kwargs):
    """
    :param inputs:        a list of Tensor contains inputs
    :param module:        a str of the python module
    :param op:            a str of the operator class
    :param param_str:     a str of param_str to be used in operator class
    :param nout:          a int of returned tensors
    :return:
    """

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(op_type='Template', **kwargs)


def Copy(inputs, **kwargs):

    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('Copy Operator accpets a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    kwargs['existing_outputs'] = [kwargs['inputs'][1]]
    kwargs['inputs'] = [kwargs['inputs'][0]]

    output =  Tensor.CreateOperator(nout=1, op_type='Copy', **kwargs)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def Accuracy(inputs, top_k=1, ignore_labels=[], **kwargs):
    """
    :param inputs:          a list of Tensor contains [input, label]
    :param top_k:           a int of the top-k accuracy
    :param ignore_labels:   a list of int contatins the labels to ignore
    :return:                a Tensor of loss with the shape (1,)
    """

    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('Accuracy Operator accpets a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output =  Tensor.CreateOperator(nout=1, op_type='Accuracy', **kwargs)
    output.shape = [1]
    return output


def StopGradient(inputs, **kwargs):
    """
    :param inputs:          a Tensor with any shape
    :return:                a Tensor, the same as inputs, but will stop gradient
    """

    if not isinstance(inputs, Tensor):
        raise RuntimeError('StopGradient Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output =  Tensor.CreateOperator(nout=1, op_type='StopGradient', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def OneHot(inputs, depth, on_value=1, off_value=0, **kwargs):
    """
    :param inputs:          a Tensor with any shape
    :param depth:           a int defining the depth of the one hot dimension
    :param on_value:        a int scalar defining the value to fill when indices[j] = i
    :param off_value:       a int scalar defining the value to fill when indices[j] != i
    :return:                a one-hot Tensor
    """

    if not isinstance(inputs, Tensor):
        raise RuntimeError('OneHot Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='OneHot', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        output.shape.append(depth)

    return output


def MovingAverage(inputs, decay, **kwargs):

    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('MovingAverage Operator accpets a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    variable = kwargs['inputs'][0]; del kwargs['inputs'][0]

    output = Tensor.CreateOperator(op_type='MovingAverage',
                                   existing_outputs=variable, **kwargs)

    return output


def Equal(inputs, **kwargs):

    if not isinstance(inputs, list) or len(inputs) is not 2:
        raise RuntimeError('Equal Operator accpets a list of 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Compare', operation='EQUAL',  **kwargs)

    if all(input.shape is not None for input in inputs):
        output.shape = inputs[0].shape[:]

    return output


def Proposal(inputs, ratio, scale,
             base_size=16, min_size=16, feat_stride=16,
             pre_nms_topn=12000, post_nms_topn=2000,
             nms_thresh=0.7, **kwargs):

    if not isinstance(inputs, list) or len(inputs) is not 3:
        raise RuntimeError('Proposal Operator accpets a list of 3 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output = Tensor.CreateOperator(nout=1, op_type='Proposal', **kwargs)

    return output