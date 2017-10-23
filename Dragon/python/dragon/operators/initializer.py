# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from . import *

def Fill(shape, value=0, **kwargs):
    """Return a Tensor with specific value filled.

    Parameters
    ----------
    shape : list, tuple or Tensor
        The shape of the new tensor.
    value : basic numerical type
        The value of the new tensor.

    Returns
    -------
    Tensor
        The value-filled Tensor.

    """
    arguments = ParseArguments(locals())
    arguments['value'] = float(value)
    if not isinstance(shape, Tensor):
        arguments['static_shape'] = shape
    else:
        arguments['dynamic_shape'] = shape.name
        arguments['extra_inputs'] = shape
    del arguments['shape']

    output =  Tensor.CreateOperator([], nout=1, op_type='Fill', **arguments)
    output.shape = arguments['static_shape'] if 'static_shape' in arguments else None
    return output


def RandomUniform(shape, low=-1.0, high=1.0, **kwargs):
    """Return a Tensor randomly initialized with uniform distribution.

    Parameters
    ----------
    shape : list, tuple or Tensor
        The shape of the new tensor.
    low : basic numerical type
        The lower bound of uniform distribution.
    high : basic numerical type
        The higher bound of uniform distribution.

    Returns
    -------
    Tensor
        The random-initialized Tensor.

    """
    arguments = ParseArguments(locals())
    arguments['low'] = float(low)
    arguments['high'] = float(high)
    if not isinstance(shape, Tensor):
        arguments['static_shape'] = shape
    else:
        arguments['dynamic_shape'] = shape.name
        arguments['extra_inputs'] = shape
    del arguments['shape']

    output =  Tensor.CreateOperator([], nout=1, op_type='RandomUniform', **arguments)
    output.shape = arguments['static_shape'] if 'static_shape' in arguments else None
    return output


def RandomNormal(shape, mean=0.0, std=1.0, **kwargs):
    """Return a Tensor randomly initialized with normal distribution.

    Parameters
    ----------
    shape : list, tuple or Tensor
        The shape of the new tensor.
    mean : basic numerical type
        The mean(mu) of normal distribution.
    std : basic numerical type
        The std(sigma) of normal distribution.

    Returns
    -------
    Tensor
        The random-initialized Tensor.

    """
    arguments = ParseArguments(locals())
    arguments['mean'] = float(mean)
    arguments['std'] = float(std)
    if not isinstance(shape, Tensor):
        arguments['static_shape'] = shape
    else:
        arguments['dynamic_shape'] = shape.name
        arguments['extra_inputs'] = shape
    del arguments['shape']

    output = Tensor.CreateOperator([], nout=1, op_type='RandomNormal', **arguments)
    output.shape = arguments['static_shape'] if 'static_shape' in arguments else None
    return output


def TruncatedNormal(shape, mean=0.0, std=1.0, **kwargs):
    """Return a Tensor randomly initialized with truncated normal distribution.

    The bounds of truncated distribution are |truncated_normal_bounds|.

    Parameters
    ----------
    shape : list, tuple or Tensor
        The shape of the new tensor.
    mean : basic numerical type
        The mean(mu) of normal distribution.
    std : basic numerical type
        The std(sigma) of normal distribution.

    Returns
    -------
    Tensor
        The random-initialized Tensor.

    """
    arguments = ParseArguments(locals())
    arguments['mean'] = float(mean)
    arguments['std'] = float(std)
    arguments['low'] = float(mean - 2.0 * std)
    arguments['high'] = float(mean + 2.0 * std)
    if not isinstance(shape, Tensor):
        arguments['static_shape'] = shape
    else:
        arguments['dynamic_shape'] = shape.name
        arguments['extra_inputs'] = shape
    del arguments['shape']

    output =  Tensor.CreateOperator([], nout=1, op_type='TruncatedNormal', **arguments)
    output.shape = arguments['static_shape'] if 'static_shape' in arguments else None
    return output


def GlorotUniform(shape, scale=3.0, mode='FAN_IN', **kwargs):
    """Return a Tensor randomly initialized with Xavier uniform distribution.

    The bounds of uniform distribution are |glorot_uniform_bounds|.

    Parameters
    ----------
    shape : list, tuple or Tensor
        The shape of the new tensor.
    scale : basic numerical type
        The scale of xavier uniform distribution.
    mode : str
        The mode, ``FAN_IN``, ``FAN_OUT`` or ``FAN_AVG``.

    Returns
    -------
    Tensor
        The random-initialized Tensor.

    """
    arguments = ParseArguments(locals())
    arguments['scale'] = float(scale)
    arguments['mode'] = mode.lower()
    if not isinstance(shape, Tensor):
        arguments['static_shape'] = shape
    else:
        arguments['dynamic_shape'] = shape.name
        arguments['extra_inputs'] = shape
    del arguments['shape']

    output = Tensor.CreateOperator([], nout=1, op_type='GlorotUniform', **arguments)
    output.shape = arguments['static_shape'] if 'static_shape' in arguments else None
    return output


def GlorotNormal(shape, scale=2.0, mode='FAN_IN', **kwargs):
    """Return a Tensor randomly initialized with MSRA normal distribution.

    The parameters of normal distribution are |glorot_normal_parameters|.

    Parameters
    ----------
    shape : list, tuple or Tensor
        The shape of the new tensor.
    scale : basic numerical type
        The scale of msra normal distribution.
    mode : str
        The mode, ``FAN_IN``, ``FAN_OUT`` or ``FAN_AVG``.

    Returns
    -------
    Tensor
        The random-initialized Tensor.

    """
    arguments = ParseArguments(locals())
    arguments['scale'] = float(scale)
    arguments['mode'] = mode.lower()
    if not isinstance(shape, Tensor):
        arguments['static_shape'] = shape
    else:
        arguments['dynamic_shape'] = shape.name
        arguments['extra_inputs'] = shape
    del arguments['shape']

    output = Tensor.CreateOperator([], nout=1, op_type='GlorotNormal', **arguments)
    output.shape = arguments['static_shape'] if 'static_shape' in arguments else None
    return output