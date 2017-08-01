# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor
import numpy as np
dtype = np.float32

def Fill(shape, value=1.0, **kwargs):
    """

    :param shape:       the shape to fill
    :param value:       the value to fill
    :return:            a value-filled Tensor

    """

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    kwargs['value'] = float(kwargs['value'])
    if not isinstance(shape, Tensor): kwargs['static_shape'] = shape
    else:
        kwargs['dynamic_shape'] = shape.name
        kwargs['extra_inputs'] = shape
    del kwargs['shape']

    output =  Tensor.CreateOperator([], nout=1, op_type='Fill', **kwargs)
    output.shape = kwargs['static_shape'] if 'static_shape' in kwargs else None
    return output


def RandomalUniform(shape, low=-1.0, high=1.0, **kwargs):
    """

    :param shape:       the shape to fill
    :param mean:        the low_bound of a uniform distribution
    :param std:         the high_bound of a uniform distribution
    :return:            a random uniform-filled Tensor

    """

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    kwargs['low'] = float(kwargs['low'])
    kwargs['high'] = float(kwargs['high'])
    if not isinstance(shape, Tensor): kwargs['static_shape'] = shape
    else:
        kwargs['dynamic_shape'] = shape.name
        kwargs['extra_inputs'] = shape
    del kwargs['shape']

    output =  Tensor.CreateOperator([], nout=1, op_type='RandomUniform', **kwargs)
    output.shape = kwargs['static_shape'] if 'static_shape' in kwargs else None
    return output


def RandomalNormal(shape, mean=0.0, std=1.0, **kwargs):
    """

    :param shape:       the shape to fill
    :param mean:        the mean of a normal distribution
    :param std:         the std of a normal distribution
    :return:            a random normal-filled Tensor

    """

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    kwargs['mean'] = float(kwargs['mean'])
    kwargs['std'] = float(kwargs['std'])
    if not isinstance(shape, Tensor): kwargs['static_shape'] = shape
    else:
        kwargs['dynamic_shape'] = shape.name
        kwargs['extra_inputs'] = shape
    del kwargs['shape']

    output = Tensor.CreateOperator([], nout=1, op_type='RandomNormal', **kwargs)
    output.shape = kwargs['static_shape'] if 'static_shape' in kwargs else None
    return output


def TruncatedNormal(shape, mean=0.0, std=1.0, **kwargs):
    """

    :param shape:       the shape to fill
    :param mean:        the mean of a normal distribution
    :param std:         the std of a normal distribution
    :return:            a truncated normal-filled Tensor

    """

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    kwargs['low'] = float(mean - 2.0 * std)
    kwargs['high'] = float(mean + 2.0 * std)
    if not isinstance(shape, Tensor): kwargs['static_shape'] = shape
    else:
        kwargs['dynamic_shape'] = shape.name
        kwargs['extra_inputs'] = shape
    del kwargs['shape']

    output =  Tensor.CreateOperator([], nout=1, op_type='TruncatedNormal', **kwargs)
    output.shape = kwargs['static_shape'] if 'static_shape' in kwargs else None
    return output


def GlorotUniform(shape, scale=3.0, mode='fan_in', **kwargs):
    """

    :param shape:       the shape to fill
    :param scale:       scaling factor (positive float)
    :param mode:        one of "fan_in", "fan_out", "fan_avg"
    :return:            a glorot uniform-filled Tensor

    """

    args = locals(); kwargs = args['kwargs']
    del args['kwargs'];  kwargs = dict(args, **kwargs)
    if not isinstance(shape, Tensor): kwargs['static_shape'] = shape
    else:
        kwargs['dynamic_shape'] = shape.name
        kwargs['extra_inputs'] = shape
    del kwargs['shape']

    output = Tensor.CreateOperator([], nout=1, op_type='GlorotUniform', **kwargs)
    output.shape = kwargs['static_shape'] if 'static_shape' in kwargs else None
    return output


def GlorotNormal(shape, scale=2.0, mode='fan_in', **kwargs):
    """

    :param shape:       the shape to fill
    :param scale:       scaling factor (positive float)
    :param mode:        one of "fan_in", "fan_out", "fan_avg"
    :return:            a glorot uniform-filled Tensor

    """

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    if not isinstance(shape, Tensor):
        kwargs['static_shape'] = shape
    else:
        kwargs['dynamic_shape'] = shape.name
        kwargs['extra_inputs'] = shape
    del kwargs['shape']

    output = Tensor.CreateOperator([], nout=1, op_type='GlorotNormal', **kwargs)
    output.shape = kwargs['static_shape'] if 'static_shape' in kwargs else None
    return output



