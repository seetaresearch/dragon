# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""The init ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.autograph.tensor import Tensor
from dragon.core.eager import context
from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import ops
from dragon.core.framework import types
from dragon.core.ops import init_ops_lib
from dragon.core.ops.utils import ArgHelper
from dragon.core.ops.utils import OpSchema
from dragon.core.ops.utils import parse_args


def constant(value, dtype=None, shape=None, name=None):
    r"""Return a tensor initialized from the value.

    Examples:

    ```python
    a = dragon.constant(1)
    b = dragon.constant(1, dtype='float32', shape=[1, 1, 1])
    c = dragon.constant(numpy.ones((2, 3))
    ```

    Parameters
    ----------
    value : array_like
        The value to initialize from.
    dtype : str, optional
        The optional data type.
    shape : Sequence[int], optional
        The optional tensor shape.
    name : str, optional
        The optional tensor name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if types.is_eager_tensor(value):
        value = value.numpy(True)
        if dtype is not None:
            value = value.astype(dtype)
    else:
        value = numpy.array(value, dtype=dtype)
    value = value.reshape(shape) if shape else value
    if context.executing_eagerly():
        return EagerTensor(value)
    else:
        return Tensor.convert_to(value, str(value.dtype), name)


def eye(n, m=None, k=0, dtype='float32', **kwargs):
    """Return a tensor constructed as the identity matrix.

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
    n : Union[int, dragon.Tensor]
        The number output rows.
    m : Union[int, dragon.Tensor], optional
        The number output cols.
    k : int, optional, default=0
        The index of diagonal.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    m = n if m is None else m
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.Eye
    if context.executing_eagerly():
        if types.is_tensor(n):
            n = int(n.get_value())
        if types.is_tensor(m):
            m = int(m.get_value())
        return op_lib \
            .instantiate(
                k=k,
                ndim=2,
                dtype=dtype,
            ).apply([n, m], trainable=trainable)
    else:
        args['n'] = args['m'] = None
        if types.is_tensor(n) or types.is_tensor(m):
            n = ops.scalar_to_tensor(n, 'int64')
            m = ops.scalar_to_tensor(m, 'int64')
            args['dims_descs'] = [n.id, m.id]
            args['extra_inputs'] = [n, m]
        else:
            args['dims'] = [n, m]
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def eye_like(other, k=0, dtype='float32', **kwargs):
    """Return a tensor of identity matrix with shape as the other.

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
    other : dragon.Tensor
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
    args = parse_args(locals())
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.Eye
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                k=k,
                dtype=dtype,
            ).apply(
                shape=[],
                shape_like=other,
                trainable=trainable,
            )
    else:
        args.pop('other')
        return op_lib.blend(inputs=[other], **args)


@ArgHelper.repeated_desc(name='shape', name_v2='dims')
def fill(shape, value=0, dtype=None, **kwargs):
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
    args = parse_args(locals())
    args['value'] = float(value)
    if dtype is None:
        args['dtype'] = str(numpy.array(value).dtype)
        if dtype == numpy.int64:
            args['dtype'] = 'int32'
        elif dtype == numpy.float64:
            args['dtype'] = 'float32'
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.Fill
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(args['dims']),
                value=args['value'],
                dtype=args['dtype'],
            ).apply(args['dims'], trainable=trainable)
    else:
        return op_lib.blend(**args)


@ArgHelper.repeated_desc(name='shape', name_v2='dims')
def glorot_normal(shape, scale=2.0, mode='fan_in', dtype='float32', **kwargs):
    r"""Return a tensor initialized from the glorot normal distribution.

    .. math:: \text{out} \sim \mathcal{N}(0, \sqrt{\frac{scale}{\text{fan}}})

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
    args = parse_args(locals())
    args['scale'] = float(scale)
    args['mode'] = mode.lower()
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.GlorotNormal
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(args['dims']),
                scale=args['scale'],
                mode=args['mode'],
                dtype=dtype,
            ).apply(args['dims'], trainable=trainable)
    else:
        return op_lib.blend(**args)


@ArgHelper.repeated_desc(name='shape', name_v2='dims')
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
    args = parse_args(locals())
    args['scale'] = float(scale)
    args['mode'] = mode.lower()
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.GlorotUniform
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(args['dims']),
                scale=args['scale'],
                mode=args['mode'],
                dtype=dtype,
            ).apply(args['dims'], trainable=trainable)
    else:
        return op_lib.blend(**args)


@ArgHelper.repeated_desc(name='shape', name_v2='dims')
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


def ones_like(other, dtype='float32', **kwargs):
    r"""Return a tensor of ones with shape as the other.

    .. math:: \text{out} \leftarrow 1

    Examples:

    ```python
    x = dragon.ones(shape=(2, 3))
    y = dragon.ones_like(x)
    ```

    Parameters
    ----------
    other : dragon.Tensor
        The tensor to hint the shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.Fill
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                value=1,
                dtype=dtype,
            ).apply(
                shape=[],
                shape_like=other,
                trainable=trainable,
            )
    else:
        args.pop('other')
        return op_lib.blend(inputs=[other], value=1., **args)


@ArgHelper.repeated_desc(name='shape', name_v2='dims')
def random_normal(shape, mean=0, std=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the normal distribution.

    .. math:: \text{out} \sim \mathcal{N}(\mu, \sigma)

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
    args = parse_args(locals())
    args['mean'] = float(mean)
    args['std'] = float(std)
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.RandomNormal
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(args['dims']),
                mean=args['mean'],
                std=args['std'],
                dtype=dtype,
            ).apply(args['dims'], trainable=trainable)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def random_normal_like(other, mean=0, std=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the normal distribution with shape as the other.

    .. math:: \text{out} \sim \mathcal{N}(\mu, \sigma)

    Parameters
    ----------
    other : dragon.Tensor
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
    args = parse_args(locals())
    args['mean'] = float(mean)
    args['std'] = float(std)
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.RandomNormal
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                mean=args['mean'],
                std=args['std'],
                dtype=dtype,
            ).apply(
                shape=[],
                shape_like=other,
                trainable=trainable,
            )
    else:
        args.pop('other')
        return op_lib.blend(inputs=[other], **args)


@ArgHelper.repeated_desc(name='shape', name_v2='dims')
def random_uniform(shape, low=-1, high=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the uniform distribution.

    .. math:: \text{out} \sim \mathcal{U}(\alpha, \beta)

    Parameters
    ----------
    shape : Sequence[Union[int, dragon.Tensor]]
        The tensor shape.
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
    args = parse_args(locals())
    args['low'], args['high'] = float(low), float(high)
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.RandomUniform
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(args['dims']),
                low=args['low'],
                high=args['high'],
                dtype=dtype,
            ).apply(args['dims'], trainable=trainable)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def random_uniform_like(other, low=-1, high=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the uniform distribution with shape as the other.

    .. math:: \text{out} \sim \mathcal{U}(\alpha, \beta)

    Parameters
    ----------
    other : dragon.Tensor
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
    args = parse_args(locals())
    args['low'], args['high'] = float(low), float(high)
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.RandomUniform
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                low=args['low'],
                high=args['high'],
                dtype=dtype,
            ).apply(
                shape=[],
                shape_like=other,
                trainable=trainable,
            )
    else:
        args.pop('other')
        return op_lib.blend(inputs=[other], **args)


@ArgHelper.repeated_desc(name='shape', name_v2='dims')
def truncated_normal(shape, mean=0, std=1, dtype='float32', **kwargs):
    r"""Return a tensor initialized from the truncated normal distribution.

    .. math:: \text{out} \sim \mathcal{TN}(\mu, \sigma, \mu - 2\sigma, \mu + 2\sigma)

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
    args = parse_args(locals())
    args['mean'], args['std'] = float(mean), float(std)
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.TruncatedNormal
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                ndim=len(args['dims']),
                mean=args['mean'],
                std=args['std'],
                dtype=dtype,
            ).apply(args['dims'], trainable=trainable)
    else:
        return op_lib.blend(**args)


@ArgHelper.repeated_desc(name='shape', name_v2='dims')
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
def zeros_like(other, dtype='float32', **kwargs):
    r"""Return a tensor of zeros with shape as the other.

    .. math:: \text{out} \leftarrow 0

    Examples:

    ```python
    x = dragon.zeros(shape=(2, 3))
    y = dragon.zeros_like(x)
    ```

    Parameters
    ----------
    other : dragon.Tensor
        The tensor to hint the shape.
    dtype : str, optional, default='float32'
        The optional data type.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = parse_args(locals())
    trainable = args.pop('trainable') if 'trainable' in args else False
    op_lib = init_ops_lib.Fill
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                value=0,
                dtype=dtype,
            ).apply(
                shape=[],
                shape_like=other,
                trainable=trainable,
            )
    else:
        args.pop('other')
        return op_lib.blend(inputs=[other], value=0., **args)
