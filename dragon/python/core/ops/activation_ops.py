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

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema


@OpSchema.num_inputs(1)
@OpSchema.convert_arg('ratio', as_target=False)
def dropout(inputs, ratio=0.5, inplace=False, **kwargs):
    r"""Set the elements of input to zero randomly.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    The **Dropout** function is defined as:

    .. math:: \text{Dropout}(x) = x * (r \sim \mathcal{B}(1, 1 - \text{ratio}))

    Examples:

    ```python
    x = dragon.ones((2, 3), 'float32')
    print(dragon.nn.dropout(x, ratio=0.5))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    ratio : Union[float, dragon.Tensor], optional, default=0.5
        The probability to zero an element.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'Dropout', inputs, outputs=inputs if inplace else [None],
            ratio=args['ratio'])
    args.pop('inplace')
    return OpLib.add('Dropout', **args)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg('ratio', as_target=False)
def drop_block(
    inputs,
    ratio=0.1,
    block_size=1,
    data_format='NCHW',
    inplace=False,
    **kwargs
):
    r"""Set the blocks over input to zero randomly.
    `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

    The **DropBlock** function is defined as:

    .. math::
        \text{DropBlock}(x_{ijk}) =
            x_{ijk} * (r_{ik} \sim \mathcal{B}(1, 1 - \gamma)) \\ \quad \\
        \text{where}\quad \gamma =
            \frac{\text{ratio}}{\text{block\_size}^{n}}
            \frac{\text{feat\_size}^{n}}{(\text{feat\_size} - \text{block\_size} + 1)^n}

    Examples:

    ```python
    x = dragon.ones((1, 3, 5, 5), 'float32')
    print(dragon.nn.drop_block(x, ratio=0.5, block_size=3))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    ratio : Union[float, dragon.Tensor], optional, default=0.1
        The probability to zero a block.
    block_size : int, optional, default=7
        The spatial block size.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'DropBlock', inputs, outputs=inputs if inplace else [None],
            block_size=block_size, data_format=data_format,
            ratio=args['ratio'])
    args.pop('inplace')
    return OpLib.add('DropBlock', **args)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg('ratio', as_target=False)
def drop_path(inputs, ratio=0.2, inplace=False, **kwargs):
    r"""Set the examples over the input to zero randomly.
    `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

    The **DropPath** function is defined as:

    .. math:: \text{DropPath}(x_{ij}) = x_{ij} * (r_{i} \sim \mathcal{B}(1, 1 - \text{ratio}))

    Examples:

    ```python
    x = dragon.ones((5, 2), 'float32')
    print(dragon.nn.drop_path(x, ratio=0.5))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    ratio : Union[float, dragon.Tensor], optional, default=0.2
        The probability to zero an example.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if context.executing_eagerly():
        return OpLib.execute(
            'DropPath', inputs, outputs=inputs if inplace else [None],
            ratio=args['ratio'])
    args.pop('inplace')
    return OpLib.add('DropPath', **args)


@OpSchema.num_inputs(1)
def elu(inputs, alpha=1.0, inplace=False, **kwargs):
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
    x = dragon.constant([-1., 0., 1.])
    print(dragon.nn.elu(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    alpha : float, optional, default=1.0
        The value to :math:`\alpha`.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    alpha = float(alpha)
    if context.executing_eagerly():
        return OpLib.execute(
            'Elu', inputs, outputs=inputs if inplace else [None], alpha=alpha)
    return OpLib.add('Elu', inputs, alpha=alpha, **kwargs)


@OpSchema.num_inputs(1)
def gelu(inputs, approximate=False, **kwargs):
    r"""Apply the gaussian error linear unit.
    `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

    The **GELU** function is defined as:

    .. math:: \text{GELU}(x) = x\cdot\frac{1}{2}[1 + \text{erf}(x / \sqrt{2})]

    Examples:

    ```python
    x = dragon.constant([-1., 0., 1.])
    print(dragon.nn.gelu(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    approximate : bool, optional, default=False
        Whether to approximate the computation.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('Gelu', inputs, approximate=approximate)
    return OpLib.add('Gelu', inputs, approximate=approximate, **kwargs)


@OpSchema.num_inputs(1)
def hardsigmoid(inputs, alpha=0.2, beta=0.5, inplace=False, **kwargs):
    r"""Apply the hard sigmoid function.

    The **HardSigmoid** function is defined as:

    .. math:: \text{HardSigmoid}(x) = \max(0, \min(1, \alpha * x + \beta))

    Examples:

    ```python
    x = dragon.constant([-2.5, -1., 0., 1., 2.5])
    print(dragon.nn.hardsigmoid(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    alpha : float, optional, default=0.2
        The value to :math:`\alpha`.
    beta : float, optional, default=0.5
        The value to :math:`\beta`.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    alpha, beta = float(alpha), float(beta)
    if context.executing_eagerly():
        return OpLib.execute(
            'HardSigmoid', inputs, outputs=inputs if inplace else [None],
            alpha=alpha, beta=beta)
    return OpLib.add('HardSigmoid', inputs, alpha=alpha, beta=beta, **kwargs)


@OpSchema.num_inputs(1)
def hardswish(inputs, **kwargs):
    r"""Apply the hard swish function.
    `[Howard et.al, 2019] <https://arxiv.org/abs/1905.02244>`_.

    The **HardSwish** function is defined as:

    .. math:: \text{HardSwish}(x) = x \cdot \max(0, \min(1, \frac{x}{6} + 0.5))

    Examples:

    ```python
    x = dragon.constant([-3., -1., 0., 1., 3.])
    print(dragon.nn.hardswish(x))
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
    if context.executing_eagerly():
        return OpLib.execute('HardSwish', inputs)
    return OpLib.add('HardSwish', inputs, **kwargs)


@OpSchema.num_inputs(1)
def leaky_relu(inputs, alpha=0.2, inplace=False, **kwargs):
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
    x = dragon.constant([-1., 0., 1.])
    print(dragon.nn.leaky_relu(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    alpha : number, optional, default=0.2
        The value to :math:`\alpha`.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    alpha = float(alpha)
    if context.executing_eagerly():
        return OpLib.execute(
            'Relu', inputs, outputs=inputs if inplace else [None], alpha=alpha)
    return OpLib.add('Relu', inputs, alpha=alpha, **kwargs)


@OpSchema.num_inputs(1)
def log_softmax(inputs, axis=-1, inplace=False, **kwargs):
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
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'LogSoftmax', inputs,
            outputs=inputs if inplace else [None], axis=axis)
    return OpLib.add('LogSoftmax', inputs, axis=axis, **kwargs)


@OpSchema.num_inputs(2)
def prelu(inputs, data_format='NCHW', **kwargs):
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
    x = dragon.constant([[-1., 0., 1.]])
    w = dragon.fill((3,), value=0.25, dtype=x.dtype)
    print(dragon.nn.prelu([x, w]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The input and weight.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('PRelu', inputs, data_format=data_format)
    return OpLib.add('PRelu', inputs, data_format=data_format, **kwargs)


@OpSchema.num_inputs(1)
def relu(inputs, inplace=False, **kwargs):
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
    print(dragon.nn.relu(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Relu', inputs, outputs=inputs if inplace else [None], alpha=0.0)
    return OpLib.add('Relu', inputs, alpha=0.0, **kwargs)


@OpSchema.num_inputs(1)
def relu6(inputs, inplace=False, **kwargs):
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
    x = dragon.constant([-1., 0., 7.])
    print(dragon.nn.relu6(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Relu', inputs, outputs=inputs if inplace else [None],
            alpha=0.0, max_value=6.0)
    return OpLib.add('Relu', inputs, alpha=0.0, max_value=6.0, **kwargs)


@OpSchema.num_inputs(1)
def selu(inputs, alpha=1.67326, gamma=1.0507, inplace=False, **kwargs):
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
    print(dragon.nn.selu(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    alpha : float, optional, default=1.67326
        The value to :math:`\alpha`.
    gamma : float, optional, default=1.0507
        The value to :math:`\gamma`.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    alpha, gamma = float(alpha), float(gamma)
    if context.executing_eagerly():
        return OpLib.execute(
            'Selu', inputs, outputs=inputs if inplace else [None],
            alpha=alpha, gamma=gamma)
    return OpLib.add('Selu', inputs, alpha=alpha, gamma=gamma, **kwargs)


@OpSchema.num_inputs(1)
def sigmoid(inputs, inplace=False, **kwargs):
    r"""Compute the sigmoid result of input.

    The **Sigmoid** function is defined as:

    .. math:: \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    x = dragon.constant([0.2, 0.4, 0.6, 0.8, 1.0])
    print(dragon.math.sigmoid(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Sigmoid', inputs, outputs=inputs if inplace else [None])
    return OpLib.add('Sigmoid', inputs, **kwargs)


@OpSchema.num_inputs(1)
def silu(inputs, **kwargs):
    r"""Apply the sigmoid linear unit.
    `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

    The **SiLU** function is defined as:

    .. math:: \text{SiLU}(x) = x \cdot \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    x = dragon.constant([-2.5, -1.0, 0.0, 1.0, 2.5])
    print(dragon.nn.silu(x))
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
    if context.executing_eagerly():
        return OpLib.execute('Silu', inputs)
    return OpLib.add('Silu', inputs, **kwargs)


@OpSchema.num_inputs(1)
def softmax(inputs, axis=-1, inplace=False, **kwargs):
    r"""Compute the softmax result.

    The **Softmax** function is defined as:

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_{i})}{\sum_{j} \exp(x_{j})}

    The argument ``axis`` could be negative:

    ```python
    x = dragon.ones((1, 4), dtype='float32')
    print(dragon.nn.softmax(x, 1))  # [[0.25 0.25 0.25 0.25]]
    print(dragon.nn.softmax(x, -1))  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    axis : int, optional, default=-1
        The axis to reduce.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Softmax', inputs, outputs=inputs if inplace else [None], axis=axis)
    return OpLib.add('Softmax', inputs, axis=axis, **kwargs)


@OpSchema.num_inputs(1)
def tanh(inputs, inplace=False, **kwargs):
    r"""Compute the tanh of input.

    The **Tanh** function is defined as:

    .. math:: \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}

    Examples:

    ```python
    x = dragon.constant([0.2, 0.4, 0.6, 0.8, 1.0])
    print(dragon.math.tanh(x))
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute(
            'Tanh', inputs, outputs=inputs if inplace else [None])
    return OpLib.add('Tanh', inputs, **kwargs)
