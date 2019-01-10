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


@ArgumentHelper.RepeatedDesc(name='shape', name_v2='dims')
def Fill(shape, value=0, dtype='float32', **kwargs):
    """Return a Tensor with specific value filled.

    **Type Constraints**: (*bool*, *int8*, *uint8*, *int32*, *int64*, *float16*, *float32*, *float64*)

    Parameters
    ----------
    shape : sequence of (int, Tensor)
        The output shape.
    value : number, optional
        The value to fill.
    dtype : str, optional
        The optional data type.

    Returns
    -------
    Tensor
        A tensor filled with the constants.

    """
    arguments = ParseArgs(locals())
    arguments['value'] = float(value)
    return Tensor.CreateOperator('Fill', [], **arguments)


@ArgumentHelper.RepeatedDesc(name='shape', name_v2='dims')
def RandomUniform(shape, low=-1.0, high=1.0, **kwargs):
    """Return a Tensor randomly initialized with *Uniform* distribution.

    **Type Constraints**: *float32*

    Parameters
    ----------
    shape : sequence of (int, Tensor)
        The shape of the new tensor.
    low : number, optional
        The lower bound of uniform distribution.
    high : number, optional
        The higher bound of uniform distribution.

    Returns
    -------
    Tensor
        A randomly initialized tensor.

    """
    arguments = ParseArgs(locals())
    arguments['low'], arguments['high'] = float(low), float(high)
    return Tensor.CreateOperator('RandomUniform', [], **arguments)


@ArgumentHelper.RepeatedDesc(name='shape', name_v2='dims')
def RandomNormal(shape, mean=0.0, std=1.0, **kwargs):
    """Return a Tensor randomly initialized with *Normal* distribution.

    **Type Constraints**: *float32*

    Parameters
    ----------
    shape : sequence of (int, Tensor)
        The shape of the new tensor.
    mean : number, optional
        The mean(mu) of normal distribution.
    std : number, optional
        The std(sigma) of normal distribution.

    Returns
    -------
    Tensor
        A randomly initialized tensor.

    """
    arguments = ParseArgs(locals())
    arguments['mean'], arguments['std'] = float(mean), float(std)
    return Tensor.CreateOperator('RandomNormal', [], **arguments)


@ArgumentHelper.RepeatedDesc(name='shape', name_v2='dims')
def TruncatedNormal(shape, mean=0.0, std=1.0, **kwargs):
    """Return a Tensor randomly initialized with *Truncated Normal* distribution.

    The bounds of truncated distribution are |truncated_normal_bounds|.

    **Type Constraints**: *float32*

    Parameters
    ----------
    shape : sequence of (int, Tensor)
        The shape of the new tensor.
    mean : number, optional
        The mean(mu) of normal distribution.
    std : number, optional
        The std(sigma) of normal distribution.

    Returns
    -------
    Tensor
        A randomly initialized tensor.

    """
    arguments = ParseArgs(locals())
    arguments['mean'] = float(mean)
    arguments['std'] = float(std)
    arguments['low'] = float(mean - 2.0 * std)
    arguments['high'] = float(mean + 2.0 * std)
    return Tensor.CreateOperator('TruncatedNormal', [], **arguments)


@ArgumentHelper.RepeatedDesc(name='shape', name_v2='dims')
def GlorotUniform(shape, scale=3.0, mode='FAN_IN', **kwargs):
    """Return a Tensor randomly initialized with *Xavier Uniform* distribution.

    The bounds of uniform distribution are |glorot_uniform_bounds|.

    **Type Constraints**: *float32*

    Parameters
    ----------
    shape : sequence of (int, Tensor)
        The shape of the new tensor.
    scale : number, optional
        The scale of xavier uniform distribution.
    mode : {'FAN_IN', 'FAN_OUT', 'FAN_AVG'}, optional
        The mode to compute the normalizer.

    Returns
    -------
    Tensor
        A randomly initialized tensor.

    """
    arguments = ParseArgs(locals())
    arguments['scale'] = float(scale)
    arguments['mode'] = mode.lower()
    return Tensor.CreateOperator('GlorotUniform', [], **arguments)


@ArgumentHelper.RepeatedDesc(name='shape', name_v2='dims')
def GlorotNormal(shape, scale=2.0, mode='FAN_IN', **kwargs):
    """Return a Tensor randomly initialized with *Kaiming Normal* distribution.

    The parameters of normal distribution are |glorot_normal_parameters|.

    **Type Constraints**: *float32*

    Parameters
    ----------
    shape : list, tuple or Tensor
        The shape of the new tensor.
    scale : number, optional
        The scale of msra normal distribution.
    mode : {'FAN_IN', 'FAN_OUT', 'FAN_AVG'}, optional
        The mode to compute the normalizer.

    Returns
    -------
    Tensor
        A randomly initialized tensor.

    """
    arguments = ParseArgs(locals())
    arguments['scale'] = float(scale)
    arguments['mode'] = mode.lower()
    return Tensor.CreateOperator('GlorotNormal', [], **arguments)