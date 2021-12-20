# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Random ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema


@OpSchema.convert_arg(name='shape', name_v2='dims')
def glorot_normal(shape, scale=2.0, mode='fan_in', dtype='float32', **kwargs):
    r"""Return a tensor initialized from the glorot normal distribution.

    .. math:: \text{out} \sim \mathcal{N}(0, \frac{scale}{\text{fan}})

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    mode : {'fan_in', 'fan_out', 'fan_avg'}, optional
        The mode to compute fans.
    scale : float, optional, default=2.0
        The scale factor to distribution.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    args['scale'] = float(scale)
    args['mode'] = mode.lower()
    if context.executing_eagerly():
        return OpLib.execute(
            'GlorotNormal', [], ndim=len(args['dims']), **args)
    return OpLib.add('GlorotNormal', [], **args)


@OpSchema.convert_arg(name='shape', name_v2='dims')
def glorot_uniform(shape, mode='fan_in', scale=3.0, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the glorot uniform distribution.

    .. math::
        \text{out} \sim \mathcal{U}(-\sqrt{\frac{scale}{\text{fan}}},
                                     \sqrt{\frac{scale}{\text{fan}}})

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    mode : {'fan_in', 'fan_out', 'fan_avg'}, optional
        The mode to compute fans.
    scale : float, optional, default=3.0
        The scale factor to distribution.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    args['scale'] = float(scale)
    args['mode'] = mode.lower()
    if context.executing_eagerly():
        return OpLib.execute(
            'GlorotUniform', [], ndim=len(args['dims']), **args)
    return OpLib.add('GlorotUniform', [], **args)


@OpSchema.num_inputs(1)
def multinomial(inputs, sample_size=1, **kwargs):
    """Return an index tensor sampled from the multinomial distribution.

    Examples:

    ```python
    input = dragon.math.log(dragon.constant([0.5, 0.5]))
    index = dragon.random.multinomial(input)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    sample_size : int, optional, default=1
        The number of samples.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Multinomial', inputs, sample_size=sample_size)
    return OpLib.add('Multinomial', inputs, sample_size=sample_size, **kwargs)


@OpSchema.convert_arg('limit')
def permutation(limit, dtype='int64', **kwargs):
    r"""Return a tensor with value in the permuted range.

    Set :attr:`limit` to determine a range :math:`[0, \text{limit})`:

    ```python
    x = dragon.random.permutation(4)
    ```

    Parameters
    ----------
    limit: Union[number, dragon.Tensor]
        The end of interval.
    dtype : str, optional, default='int64'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    args['dtype'] = args['dtype'].lower()
    if context.executing_eagerly():
        return OpLib.execute(
            'Permutation', [], dtype=dtype, limit=args['limit'])
    return OpLib.add('Permutation', [], **args)


@OpSchema.convert_arg(name='shape', name_v2='dims')
def random_normal(shape, mean=0, std=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the normal distribution.

    .. math:: \text{out} \sim \mathcal{N}(\mu, \sigma^{2})

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    args['mean'] = float(mean)
    args['std'] = float(std)
    if context.executing_eagerly():
        return OpLib.execute(
            'RandomNormal', [], ndim=len(args['dims']), **args)
    return OpLib.add('RandomNormal', [], **args)


@OpSchema.num_inputs(1)
def random_normal_like(inputs, mean=0, std=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the normal distribution with shape as the other.

    .. math:: \text{out} \sim \mathcal{N}(\mu, \sigma^{2})

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor to hint the shape.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    mean, std = float(mean), float(std)
    if context.executing_eagerly():
        return OpLib.execute(
            'RandomNormal', inputs, mean=mean, std=std, dtype=dtype)
    return OpLib.add('RandomNormal', inputs,
                     mean=mean, std=std, dtype=dtype, **kwargs)


@OpSchema.convert_arg(name='shape', name_v2='dims')
def random_uniform(shape, low=0, high=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the uniform distribution.

    .. math:: \text{out} \sim \mathcal{U}(\alpha, \beta)

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    low : number, optional, default=0
        The value to :math:`\alpha`.
    high : number, optional, default=1
        The value to :math:`\beta`.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    args['low'], args['high'] = float(low), float(high)
    if context.executing_eagerly():
        return OpLib.execute(
            'RandomUniform', [], ndim=len(args['dims']), **args)
    return OpLib.add('RandomUniform', [], **args)


@OpSchema.num_inputs(1)
def random_uniform_like(inputs, low=-1, high=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the uniform distribution with shape as the other.

    .. math:: \text{out} \sim \mathcal{U}(\alpha, \beta)

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor to hint the shape.
    low : number, optional, default=-1
        The value to :math:`\alpha`.
    high : number, optional, default=1
        The value to :math:`\beta`.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    low, high = float(low), float(high)
    if context.executing_eagerly():
        return OpLib.execute(
            'RandomUniform', inputs, low=low, high=high, dtype=dtype)
    return OpLib.add('RandomUniform', inputs,
                     low=low, high=high, dtype=dtype, **kwargs)


@OpSchema.convert_arg(name='shape', name_v2='dims')
def truncated_normal(shape, mean=0, std=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the truncated normal distribution.

    .. math:: \text{out} \sim \mathcal{TN}(\mu, \sigma^{2},
                                           \mu - 2\sigma, \mu + 2\sigma)

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    mean : number, optional, default=0
        The value to :math:`\mu`.
    std : number, optional, default=1
        The value to :math:`\sigma`.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    args['mean'], args['std'] = float(mean), float(std)
    if context.executing_eagerly():
        return OpLib.execute(
            'TruncatedNormal', [], ndim=len(args['dims']), **args)
    return OpLib.add('TruncatedNormal', [], **args)
