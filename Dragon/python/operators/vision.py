# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import math
from dragon.core.tensor import Tensor

def Conv2D(inputs, num_output, kernel_size,
           stride=1, pad=0, dilation=1, group=1, **kwargs):
    """

    :param inputs:      a list of Tensor contains [input, weight] or [input, weight, bias]
    :param num_output:  a int of the output feature maps
    :param kernel:      a tuple or a int of the kernel size
    :param stride:      a tuple or a int of the stride size
    :param pad:         a tuple or a int of the zero-padding size
    :param dilation:    a tuple or a int of the dilation size
    :param scale:       divide by the kernel size (theano's implemention)
    :return:            a 4D Tensor with the shape (num, channels, height, width)

    """

    if not isinstance(inputs, list) or len(inputs) < 2:
        raise TypeError('Conv2D Operator accpets a list of at least 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    if not isinstance(kwargs['kernel_size'], list):
        kwargs['kernel_size'] = [kwargs['kernel_size']]

    if not isinstance(kwargs['stride'], list):
        kwargs['stride'] = [kwargs['stride']]

    if not isinstance(kwargs['pad'], list):
        kwargs['pad'] = [kwargs['pad']]

    if not isinstance(kwargs['dilation'], list):
        kwargs['dilation'] = [kwargs['dilation']]

    output = Tensor.CreateOperator(nout=1, op_type='Conv', **kwargs)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]
        output.shape[1] = num_output
        for i in xrange(2):
            k = kwargs['kernel_size'][i] if i < len(kwargs['kernel_size']) \
                                         else kwargs['kernel_size'][-1]
            s = kwargs['stride'][i]      if i < len(kwargs['stride']) \
                                         else kwargs['stride'][-1]
            p = kwargs['pad'][i]         if i < len(kwargs['pad']) \
                                         else kwargs['pad'][-1]
            d = kwargs['dilation'][i]    if i < len(kwargs['dilation']) \
                                         else kwargs['dilation'][-1]
            dk = d * (k - 1) + 1
            output.shape[i + 2] = (output.shape[i + 2] + 2 * p - dk) / s + 1

    return output


def Deconv2D(inputs, num_output, kernel_size,
             stride=1, pad=0, dilation=1, group=1, **kwargs):

    if not isinstance(inputs, list) or len(inputs) < 2:
        raise TypeError('Deconv2D Operator accpets a list of at least 2 Tensors')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    if not isinstance(kwargs['kernel_size'], list):
        kwargs['kernel_size'] = [kwargs['kernel_size']]

    if not isinstance(kwargs['stride'], list):
        kwargs['stride'] = [kwargs['stride']]

    if not isinstance(kwargs['pad'], list):
        kwargs['pad'] = [kwargs['pad']]

    if not isinstance(kwargs['dilation'], list):
        kwargs['dilation'] = [kwargs['dilation']]

    return Tensor.CreateOperator(nout=1, op_type='DeConv', **kwargs)


def Pool2D(inputs, kernel_size, stride, pad=0, mode='MAX_POOLING', **kwargs):
    """

    :param inputs:           a 3D Tensor with [channels, height, width] or
                             a 4D Tensor with [num, channels, height, width]
    :param kernel_size:      a tuple or a int of the kernel size
    :param stride:           a tuple or a int of the stride size
    :param pad:              a tuple or a int of the zero-padding size
    :param way:              a string of 'MAX_POOLING' or 'AVG_POOLING'
    :return:                 a 3D or 4D Tensor of the pooled output

    """
    if not isinstance(inputs, Tensor):
        raise RuntimeError('Pooling Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    SUPPORT_MODES = {'MAX_POOLING': 0, 'AVG_POOLING': 1}
    kwargs['mode'] = SUPPORT_MODES[mode]

    if not isinstance(kwargs['kernel_size'], list):
        kwargs['kernel_size'] = [kwargs['kernel_size']]

    if not isinstance(kwargs['stride'], list):
        kwargs['stride'] = [kwargs['stride']]

    if not isinstance(kwargs['pad'], list):
        kwargs['pad'] = [kwargs['pad']]

    output = Tensor.CreateOperator(nout=1, op_type='Pooling', **kwargs)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        for i in xrange(2):
            k = kwargs['kernel_size'][i] if i < len(kwargs['kernel_size']) \
                                         else kwargs['kernel_size'][-1]
            s = kwargs['stride'][i]      if i < len(kwargs['stride']) \
                                         else kwargs['stride'][-1]
            p = kwargs['pad'][i]         if i < len(kwargs['pad']) \
                                         else kwargs['pad'][-1]
            output.shape[i + 2] = int(math.ceil(float(output.shape[i + 2] + 2 * p - k) / s) + 1)

    return output


def ROIPooling(inputs, pool_h=0, pool_w=0, spatial_scale=1.0, **kwargs):

    if not isinstance(inputs, list) or len(inputs) < 2:
        raise RuntimeError('ROIPooling Operator accepts 2 Tensors as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(nout=1, op_type='ROIPooling', **kwargs)


def ROIAlign(inputs, pool_h=0, pool_w=0, spatial_scale=1.0, **kwargs):

    if not isinstance(inputs, list) or len(inputs) < 2:
        raise RuntimeError('ROIAlign Operator accepts 2 Tensors as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(nout=1, op_type='ROIAlign', **kwargs)


def LRN(inputs, local_size, alpha, beta, mode='ACROSS_CHANNELS', **kwargs):

    if not isinstance(inputs, Tensor):
        raise RuntimeError('LRN Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    SUPPORT_MODES = {'ACROSS_CHANNELS': 0, 'WITHIN_CHANNEL': 1}
    kwargs['mode'] = SUPPORT_MODES[mode]

    return Tensor.CreateOperator(nout=1, op_type='LRN', **kwargs)


def NNResize(inputs, dsize=(), fy=-1.0, fx=-1.0, **kwargs):
    if not isinstance(inputs, Tensor):
        raise RuntimeError('NNResize Operator accepts a Tensor as inputs')

    if len(dsize) == 0 and (fy == -1.0 or fx==-1.0):
        raise RuntimeError('NNResize should be specified either dsize or fy/fx')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    return Tensor.CreateOperator(nout=1, op_type='NNResize', **kwargs)


def BiasAdd(inputs, data_format='NCHW', **kwargs):

    if not isinstance(inputs, list) or len(inputs) != 2:
        raise RuntimeError('BiasAdd Operator accepts 2 Tensors as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    output =  Tensor.CreateOperator(nout=1, op_type='BiasAdd', **kwargs)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output

def DenseConcat(inputs, axis=1, **kwargs):

    if not isinstance(inputs, list) or len(inputs) != 2:
        raise RuntimeError('DenseConcat Operator accepts 2 Tensors as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    kwargs['num_input'] = len(inputs)

    output = Tensor.CreateOperator(nout=1, op_type='DenseConcat', **kwargs)
    if all(input.shape is not None for input in inputs):
        if all(input.shape[axis] is not None for input in inputs):
            output.shape = inputs[0].shape[:]
            for i in xrange(1, len(inputs)):
                output.shape[axis] += inputs[i].shape[axis]

    return output