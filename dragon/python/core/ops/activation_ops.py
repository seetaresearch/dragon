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
"""Activation ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.eager import context
from dragon.core.ops import activation_ops_lib
from dragon.core.ops import math_ops
from dragon.core.ops import array_ops
from dragon.core.ops.utils import ArgHelper
from dragon.core.ops.utils import OpSchema


@OpSchema.num_inputs(1)
@ArgHelper.desc('ratio', as_target=False)
def dropout(inputs, ratio=0.5, **kwargs):
    r"""Set the elements of the input to zero randomly.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    The **Dropout** function is defined as:

    .. math:: \text{Dropout}(x) = x * (r \sim \mathcal{B}(1, 1 - \text{ratio}))

    Examples:

    ```python
    x = dragon.ones((2, 3), 'float32')
    print(dragon.nn.dropout(x, 0.5, inplace=False))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    ratio : Union[float, dragon.Tensor], optional, default=0.5
        The dropping ratio.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.Dropout
    if context.executing_eagerly():
        return op_lib \
            .instantiate() \
            .apply([inputs], args['ratio'], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
@ArgHelper.desc('ratio', as_target=False)
def drop_block2d(inputs, ratio=0.1, block_size=7, data_format='NCHW', **kwargs):
    r"""Set the spatial blocks over input to zero randomly.
    `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

    The **DropBlock** function is defined as:

    .. math::
        \text{DropBlock}(x_{ijk}) =
            x_{ijk} * (r_{ik} \sim \mathcal{B}(1, 1 - \gamma)) \\ \quad \\
                \text{where}\quad \gamma =
                    \frac{\text{ratio}}{\text{block\_size}^{n}}
                    \frac{\text{feat\_size}^{n}}{(\text{feat\_size} - \text{block\_size} + 1)^n}

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    ratio : Union[float, dragon.Tensor], optional, default=0.1
        The dropping ratio.
    block_size : int, optional, default=7
        The spatial block size.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.DropBlock2d
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                block_size=block_size,
                data_format=data_format,
            ).apply([inputs], args['ratio'], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
@ArgHelper.desc('ratio', as_target=False)
def drop_path(inputs, ratio=0.2, **kwargs):
    r"""Set the examples over the input to zero randomly.
    `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

    The **DropPath** function is defined as:

    .. math:: \text{DropPath}(x_{ij}) = x_{ij} * (r_{i} \sim \mathcal{B}(1, 1 - \text{ratio}))

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    ratio : Union[float, dragon.Tensor], optional, default=0.2
        The dropping ratio.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.DropPath
    if context.executing_eagerly():
        return op_lib \
            .instantiate() \
            .apply([inputs], args['ratio'], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def elu(inputs, alpha=1.0, **kwargs):
    r"""Apply the exponential linear unit.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    The **ELU** function is defined as:

    .. math::
        \text{ELU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                \alpha * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = dragon.constant([-1, 0, 1], 'float32')
    print(dragon.nn.elu(x, inplace=False))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    alpha : float, optional, default=1.0
        The value to :math:`\alpha`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['alpha'] = float(alpha)
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.Elu
    if context.executing_eagerly():
        return op_lib \
            .instantiate(alpha=args['alpha']) \
            .apply([inputs], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def hardsigmoid(inputs, alpha=0.2, beta=0.5, **kwargs):
    r"""Apply the hard sigmoid function.

    The **HardSigmoid** function is defined as:

    .. math:: \text{HardSigmoid}(x) = \max(0, \min(1, \alpha * x + \beta))

    Examples:

    ```python
    x = dragon.constant([-2.5, -1.0, 0.0, 1.0, 2.5])
    print(dragon.nn.hardsigmoid(x, inplace=False))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    alpha : float, optional, default=0.2
        The value to :math:`\alpha`.
    beta : float, optional, default=0.5
        The value to :math:`\beta`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['alpha'] = float(alpha)
    args['beta'] = float(beta)
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.HardSigmoid
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                alpha=args['alpha'],
                beta=args['beta'],
            ).apply([inputs], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def hardswish(inputs, alpha=0.2, beta=0.5, **kwargs):
    r"""Apply the hard swish function.
    `[Howard et.al, 2019] <https://arxiv.org/abs/1905.02244>`_.

    The **HardSwish** function is defined as:

    .. math:: \text{HardSwish}(x) = x \cdot \max(0, \min(1, \alpha * x + \beta))

    Examples:

    ```python
    x = dragon.constant([-2.5, -1.0, 0.0, 1.0, 2.5])
    print(dragon.nn.hardswish(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    alpha : float, optional, default=0.2
        The value to :math:`\alpha`.
    beta : float, optional, default=0.5
        The value to :math:`\beta`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['alpha'] = float(alpha)
    args['beta'] = float(beta)
    op_lib = activation_ops_lib.HardSwish
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                alpha=args['alpha'],
                beta=args['beta'],
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def leaky_relu(inputs, alpha=0.2, **kwargs):
    r"""Apply the leaky rectified linear unit.

    The **LeakyReLU** function is defined as:

    .. math::
        \text{LeakyReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                \alpha * x, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = dragon.constant([-1, 0, 1], 'float32')
    print(dragon.nn.leaky_relu(x, inplace=False))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    alpha : number, optional, default=0.2
        The value to :math:`\alpha`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['alpha'] = float(alpha)
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.Relu
    if context.executing_eagerly():
        return op_lib \
            .instantiate(alpha=args['alpha']) \
            .apply([inputs], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def log_softmax(inputs, axis=-1, **kwargs):
    r"""Compute the composite of logarithm and softmax.

    The **LogSoftmax** function is defined as:

    .. math:: \text{LogSoftmax}(x) = \log(\frac{\exp(x_{i})}{\sum \exp(x_{j})})

    The argument ``axis`` could be negative:

    ```python
    x = dragon.random.uniform((2, 3), -0.1, 0.1)
    print(dragon.nn.log_softmax(x, 1))
    print(dragon.nn.log_softmax(x, -1))  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=-1
        The axis to reduce.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.sub(
        [inputs, math_ops.log(array_ops.sum(
            math_ops.exp(inputs, **kwargs),
            axis=[axis], keep_dims=True, **kwargs), **kwargs)],
        **kwargs
    )


@OpSchema.num_inputs(2)
def prelu(inputs, channel_shared=False, data_format='NCHW', **kwargs):
    r"""Apply the parametric rectified linear unit.
    `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

    The **PReLU** function is defined as:

    .. math::
        \text{PReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                weight * x, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = dragon.constant([[-1, 0, 1]], 'float32')
    w = dragon.fill([3], value=0.25, dtype='float32')
    print(dragon.nn.prelu([x, w]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input and weight.
    channel_shared : bool, optional, default=False.
        Whether to share the weight across channels.
    data_format : {'NCHW', 'NHWC'}, optional
        The data format.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    op_lib = activation_ops_lib.PRelu
    if context.executing_eagerly():
        return op_lib \
            .instantiate(data_format=data_format) \
            .apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def relu(inputs, **kwargs):
    r"""Apply the rectified linear unit.
    `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    The **ReLU** function is defined as:

    .. math::
        \text{ReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                0, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = dragon.constant([-1., 0., 1.])
    print(dragon.nn.relu(x, inplace=False))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.Relu
    if context.executing_eagerly():
        return op_lib \
            .instantiate(alpha=0.,) \
            .apply([inputs], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def relu6(inputs, **kwargs):
    r"""Apply the clipped-6 rectified linear unit.
    `[Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_.

    The **ReLU-6** function is defined as:

    .. math::
        \text{ReLU-6}(x) =
            \begin{cases}
                \min(x, 6), & \text{ if } x \geq 0 \\
                0, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = dragon.constant([-1, 0, 1], 'float32')
    print(dragon.nn.relu6(x, inplace=False))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.Relu6
    if context.executing_eagerly():
        return op_lib.instantiate().apply([inputs], inplace=inplace)
    else:
        return op_lib.blend('Relu', max_value=6., **args)


@OpSchema.num_inputs(1)
def selu(inputs, alpha=1.67326, gamma=1.0507, **kwargs):
    r"""Apply the scaled exponential linear unit.
    `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    The **SELU** function is defined as:

    .. math::
        \text{SELU}(x) = \gamma *
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                \alpha * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = dragon.constant([-1., 0., 1.])
    print(dragon.nn.selu(x, inplace=False))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    alpha : float, optional, default=1.67326
        The value to :math:`\alpha`.
    gamma : float, optional, default=1.0507
        The value to :math:`\gamma`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['alpha'], args['gamma'] = float(alpha), float(gamma)
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.Selu
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                alpha=args['alpha'],
                gamma=args['gamma'],
            ).apply([inputs], inplace=inplace)
    else:
        return op_lib.blend('Selu', **args)


@OpSchema.num_inputs(1)
def sigmoid(inputs, **kwargs):
    r"""Compute the sigmoid result of input.

    The **Sigmoid** function is defined as:

    .. math:: \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    x = dragon.constant([0.2, 0.4, 0.6, 0.8, 1.0])
    print(dragon.math.sigmoid(x, inplace=False))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.Activation
    if context.executing_eagerly():
        return op_lib \
            .instantiate(op_type='Sigmoid') \
            .apply([inputs], inplace=inplace)
    else:
        return op_lib.blend('Sigmoid', **args)


@OpSchema.num_inputs(1)
def softmax(inputs, axis=-1, **kwargs):
    r"""Compute the softmax result.

    The **Softmax** function is defined as:

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_{i})}{\sum_{j} \exp(x_{j})}

    The argument ``axis`` could be negative:

    ```python
    x = dragon.ones((1, 4), dtype='float32')
    print(dragon.nn.softmax(x, 1))   # [[0.25 0.25 0.25 0.25]]
    print(dragon.nn.softmax(x, -1))  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=-1
        The axis to reduce.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.Softmax
    if context.executing_eagerly():
        return op_lib \
            .instantiate(axis=axis) \
            .apply([inputs], inplace=inplace)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def tanh(inputs, **kwargs):
    r"""Compute the tanh of input.

    The **Tanh** function is defined as:

    .. math:: \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}

    Examples:

    ```python
    x = dragon.constant([0.2, 0.4, 0.6, 0.8, 1.0], 'float32')
    print(dragon.math.tanh(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    inplace = args.pop('inplace') if 'inplace' in args else False
    op_lib = activation_ops_lib.Activation
    if context.executing_eagerly():
        return op_lib \
            .instantiate(op_type='Tanh') \
            .apply([inputs], inplace=inplace)
    else:
        return op_lib.blend('Tanh', **args)


@OpSchema.num_inputs(1)
def swish(inputs, **kwargs):
    r"""Apply the swish function.
    `[Ramachandran et.al, 2017] <https://arxiv.org/abs/1710.05941>`_.

    The **Swish** function is defined as:

    .. math:: \text{Swish}(x) = x \cdot \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    x = dragon.constant([-2.5, -1.0, 0.0, 1.0, 2.5])
    print(dragon.nn.swish(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    op_lib = activation_ops_lib.Activation
    if context.executing_eagerly():
        return op_lib \
            .instantiate(op_type='Swish') \
            .apply([inputs])
    else:
        return op_lib.blend('Swish', **args)
