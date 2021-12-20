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
"""Init ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema


def eye(n, m=None, k=0, dtype='float32', **kwargs):
    r"""Return a tensor constructed as the identity matrix.

    .. math:: \text{out} \leftarrow \text{diag}(1, 1, ..., 1)

    The rows and cols of matrix are determined by ``n`` and ``m``:

    ```python
    print(dragon.eye(2))     # [[1., 0.], [0., 1.]]
    print(dragon.eye(2, 3))  # [[1., 0., 0.], [0., 1., 0.]]
    ```

    The diagonal could be controlled by ``k``:

    * k > 0: Populate upper diagonal

    * k = 0: Populate main diagonal

    * k < 0: Populate lower diagonal

    Parameters
    ----------
    n : int
        The number of output rows.
    m : int, optional
        The number of output cols.
    k : int, optional, default=0
        The index of diagonal.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    dims = (n, n if m is None else m)
    if context.executing_eagerly():
        return OpLib.execute('Eye', [], ndim=2, dims=dims, k=k, dtype=dtype)
    return OpLib.add('Eye', [], dims=dims, k=k, dtype=dtype, **kwargs)


@OpSchema.num_inputs(1)
def eye_like(inputs, k=0, dtype='float32', **kwargs):
    r"""Return a tensor of identity matrix with shape as the other.

    .. math:: \text{out} \leftarrow \text{diag}(1, 1, ..., 1)

    The rows and cols of matrix are hinted by the input tensor:

    ```python
    x = dragon.ones(2, 3)
    print(dragon.eye_like(x))  # [[1., 0.], [0., 1.]]
    ```

    The diagonal could be controlled by ``k``:

    * k > 0: Populate upper diagonal

    * k = 0: Populate main diagonal

    * k < 0: Populate lower diagonal

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor to hint the shape.
    k : int, optional, default=0
        The index of diagonal.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Eye', inputs, k=k, dtype=dtype)
    return OpLib.add('Eye', inputs, k=k, dtype=dtype, **kwargs)


@OpSchema.convert_arg(name='shape', name_v2='dims')
def fill(shape, value=0, dtype='float32', **kwargs):
    r"""Return a tensor filled with the scalar value.

    .. math:: \text{out} \leftarrow \text{value}

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    value : number, optional, default=0
        The value to fill.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    args['value'] = float(value)
    if context.executing_eagerly():
        return OpLib.execute('Fill', [], ndim=len(args['dims']), **args)
    return OpLib.add('Fill', [], **args)


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


@OpSchema.convert_arg(name='shape', name_v2='dims')
def ones(shape, dtype='float32', **kwargs):
    r"""Return a tensor filled with ones.

    .. math:: \text{out} \leftarrow 1

    ```python
    x = dragon.ones(shape=(2, 3), dtype='float32')
    ```

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return fill(shape, 1, dtype, **kwargs)


@OpSchema.num_inputs(1)
def ones_like(inputs, dtype='float32', **kwargs):
    r"""Return a tensor of ones with shape as the other.

    .. math:: \text{out} \leftarrow 1

    Examples:

    ```python
    x = dragon.ones(shape=(2, 3))
    y = dragon.ones_like(x)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor to hint the shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Fill', inputs, value=1.0, dtype=dtype)
    return OpLib.add('Fill', inputs, value=1.0, dtype=dtype, **kwargs)


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


@OpSchema.convert_arg(name='shape', name_v2='dims')
def zeros(shape, dtype='float32', **kwargs):
    r"""Return a tensor filled with zeros.

    .. math:: \text{out} \leftarrow 0

    ```python
    x = dragon.zeros(shape=(2, 3), dtype='float32')
    ```

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return fill(shape, 0, dtype, **kwargs)


@OpSchema.num_inputs(1)
def zeros_like(inputs, dtype='float32', **kwargs):
    r"""Return a tensor of zeros with shape as the other.

    .. math:: \text{out} \leftarrow 0

    Examples:

    ```python
    x = dragon.zeros(shape=(2, 3))
    y = dragon.zeros_like(x)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The tensor to hint the shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Fill', inputs, value=0.0, dtype=dtype)
    return OpLib.add('Fill', inputs, value=0.0, dtype=dtype, **kwargs)
