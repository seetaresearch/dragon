# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
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


def _wrap_input_shape(arguments, shape):
    if isinstance(shape, Tensor):
        arguments['extra_inputs'] = shape
        arguments['shape'] = shape.name
    elif isinstance(shape, (list, tuple)):
        arguments['dims'] = shape
        arguments['shape'] = None
        AddArgumentsWithDesc(arguments, shape, 'dims', 'int32', as_target=True)
    else:
        raise TypeError('Unsupported type of shape: {}'.format(type(shape)))
    return arguments


def _wrap_output_shape(output, shape):
    if not isinstance(shape, Tensor):
        if any(isinstance(dim, Tensor) for dim in shape): return output
        output.shape = [dim for dim in shape]
    return output


def Fill(shape, value=0, **kwargs):
    """Return a Tensor with specific value filled.

    Parameters
    ----------
    shape : list, tuple or Tensor
        The output shape.
    value : basic numerical type
        The value to fill.

    Returns
    -------
    Tensor
        The constant-filled tensor.

    """
    arguments = ParseArguments(locals())
    arguments['value'] = float(value)
    arguments = _wrap_input_shape(arguments, shape)
    output =  Tensor.CreateOperator([], nout=1, op_type='Fill', **arguments)
    return _wrap_output_shape(output, shape)


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
    arguments = _wrap_input_shape(arguments, shape)
    output =  Tensor.CreateOperator([], nout=1, op_type='RandomUniform', **arguments)
    return _wrap_output_shape(output, shape)


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
    arguments = _wrap_input_shape(arguments, shape)
    output = Tensor.CreateOperator([], nout=1, op_type='RandomNormal', **arguments)
    return _wrap_output_shape(output, shape)


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
    arguments = _wrap_input_shape(arguments, shape)
    output =  Tensor.CreateOperator([], nout=1, op_type='TruncatedNormal', **arguments)
    return _wrap_output_shape(output, shape)


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
    arguments = _wrap_input_shape(arguments, shape)
    output = Tensor.CreateOperator([], nout=1, op_type='GlorotUniform', **arguments)
    return _wrap_output_shape(output, shape)


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
    arguments = _wrap_input_shape(arguments, shape)
    output = Tensor.CreateOperator([], nout=1, op_type='GlorotNormal', **arguments)
    return _wrap_output_shape(output, shape)