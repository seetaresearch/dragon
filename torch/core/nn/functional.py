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
"""NN functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest
from dragon.vm.torch.core.nn.modules import _functions
from dragon.vm.torch.core.nn import _reduction
from dragon.vm.torch.core.nn.modules import utils


def affine(input, weight, bias=None):
    r"""Apply the affine transformation to input.

    .. math:: y = Ax + b

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The optional bias.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Affine \
        .instantiate(
            input.device,
        ).apply(input, weight, bias)


def avg_pool2d(
    input,
    kernel_size,
    stride=1,
    padding=0,
    ceil_mode=False,
    global_pooling=False,
):
    r"""Apply the 2d average pooling to input.

    The spatial output dimension is computed as:

    .. math::
        \text{Dim}_{out} = (\text{Dim}_{in} +
            2 * pad - \text{K}_{size}) / stride + 1

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kernel_size : Union[int, Sequence[int]]
        The size of sliding window.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of sliding window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero-padding size.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.
    global_pooling : bool, optional
        **True** to pool globally regardless of ``kernel_size``.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _pool(
        _pool_mode='AVG',
        _nd_util=utils._pair,
        _pool_fn=_functions.Pool2d,
        **locals()
    )


def batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    r"""Apply the batch normalization to input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The moving average of stats are calculated as:

    .. math::
        x_{moving} \leftarrow (1 - momentum) * x_{moving} + momentum * x_{stat}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    running_mean : dragon.vm.torch.Tensor
        The running mean.
    running_var : dragon.vm.torch.Tensor
        The running var.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor
        The bias tensor.
    training : bool, optional, default=False
        The flag to determine the stats.
    momentum : float, optional, default=0.1
        The momentum of moving average.
    eps : float, optional, default=1e-5
        The epsilon value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.BatchNorm \
        .instantiate(
            input.device,
            training=training,
            momentum=momentum,
            eps=eps,
        ).apply(input, running_mean, running_var, weight, bias)


def binary_cross_entropy_with_logits(
    input,
    target,
    weight=None,
    size_average=None,
    reduce=None,
    reduction='mean',
    pos_weight=None,
):
    r"""Compute the sigmoid cross entropy with contiguous target.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    target : dragon.vm.torch.Tensor
        The target tensor.
    weight : dragon.vm.torch.Tensor, optional
        The weight for each class.
    size_average : bool, optional
        Whether to average the loss.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'none', 'mean', 'sum', 'valid'}, optional
        The reduce method.
    pos_weight : dragon.vm.torch.Tensor, optional
        The weight for positive examples.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return _functions.SigmoidCrossEntropy \
        .instantiate(
            input.device,
            reduction=reduction,
        ).apply([input, target])


def conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Apply the 2d convolution to input.

    The spatial output dimension is computed as:

    .. math::
        \begin{cases}
            \text{DK}_{size} = dilation *
                (\text{K}_{size} - 1) + 1 \\
            \text{Dim}_{out} = (\text{Dim}_{in} +
                2 * pad - \text{DK}_{size}) / stride + 1
        \end{cases}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The optional bias tensor.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of sliding window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero-padding size.
    dilation : Union[int, Sequence[int]], optional, default=1
        The rate of dilated kernel.
    groups : int, optional, default=1
        The number of groups to split input channels.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _conv(
        _nd_util=utils._pair,
        _conv_fn=_functions.Conv2d,
        **locals()
    )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    r"""Apply the 2d deconvolution to input.

    The spatial output dimension is computed as:

    .. math::
        \begin{cases}
            \text{DK}_{size} = dilation *
                (\text{K}_{size} - 1) + 1 \\
            \text{Dim}_{out} = (\text{Dim}_{in} - 1) *
                stride + \text{DK}_{size} - 2 * pad
        \end{cases}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The optional bias.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of sliding window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero-padding size.
    output_padding : int, optional, default=1
        The additional padding size.
    groups : int, optional, default=1
        The number of groups to split input channels.
    dilation : Union[int, Sequence[int]], optional, default=1
        The rate of dilated kernel.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _conv_transpose(
        _nd_util=utils._pair,
        _conv_fn=_functions.ConvTranspose2d,
        **locals()
    )


def cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=None,
    reduce=None,
    reduction='mean',
):
    r"""Compute the softmax cross entropy with sparse target.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    target : dragon.vm.torch.Tensor
        The target tensor.
    weight : dragon.vm.torch.Tensor, optional
        The weight for each class.
    size_average : bool, optional
        Whether to average the loss.
    ignore_index : int, optional
        The label index to ignore.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'none', 'mean', 'sum', 'valid'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The loss.

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return _functions.SparseSoftmaxCrossEntropy \
        .instantiate(
            input.device,
            reduction=reduction,
            ignore_index=ignore_index,
        ).apply([input, target])


def ctc_loss(input, target, padding_mask=-1, reduction='mean'):
    r"""Compute the ctc loss with batched labels.
    `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    target : dragon.vm.torch.Tensor
        The target tensor.
    padding_mask : int, optional, default=-1
        The mask for padding the redundant labels.
    reduction : {'none', 'mean', 'sum'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The loss.

    """
    prob = softmax(input, 2)
    return _functions.CTCLoss \
        .instantiate(
            input.device,
            reduction=reduction,
            padding_mask=padding_mask,
        ).apply([prob, target])


def depthwise_conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
):
    r"""Apply the 2d depthwise convolution to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The optional bias.
    stride : Sequence[int], default=1
        The stride of sliding window.
    padding : Sequence[int], default=0
        The zero-padding size.
    dilation : Sequence[int], default=1
        The rate of dilated kernel.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _conv(
        _nd_util=utils._pair,
        _conv_fn=_functions.DepthwiseConv2d,
        **locals()
    )


def dropout(input, p=0.5, training=True, inplace=False):
    r"""Set the elements of the input to zero randomly.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    The **Dropout** function is defined as:

    .. math:: \text{Dropout}(x) = x * \text{Bernoulli}(p=1 - prob)

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    p : float, optional, default=0.5
        The dropping prob.
    training : bool, optional, default=True
        The training flag.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if not training:
        return input
    return _functions.Dropout \
        .instantiate(
            input.device,
            p=p,
        ).apply(input, inplace=inplace)


def drop_block2d(
    input,
    kp=0.9,
    block_size=7,
    alpha=1.,
    decrement=0.,
    training=True,
    inplace=False,
    slot=None,
):
    r"""Set the spatial blocks over input to zero randomly.

    The **DropBlock** function is defined as:

    .. math::
        \text{DropBlock}(x) = x \cdot \text{Bernoulli}(\alpha\cdot\gamma) \\
        \quad \\ \text{where}\quad \gamma =
            \frac{keep\_prob}{block\_size^{n}}
            \frac{feat\_size^{n}}{(feat\_size - block\_size + 1)^n}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kp : float, optional, default=0.9
        The keeping prob.
    block_size : int, optional, default=7
        The size of a spatial block.
    alpha : float, optional, default=1.
        The scale factor to :math:`\gamma`.
    decrement : float, optional, default=0.
        The decrement value to ``kp``.
    training : bool, optional, default=True
        The training flag.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.
    slot : int, optional
        The optional slot index.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if not training:
        return input
    return _functions.DropBlock2d \
        .instantiate(
            input.device,
            keep_prob=kp,
            block_size=block_size,
            alpha=alpha,
            decrement=decrement,
            slot=slot,
        ).apply(input, inplace=inplace)


def drop_path(
    input,
    p=0.2,
    increment=0.,
    training=True,
    inplace=False,
    slot=None,
):
    r"""Set the examples over input to zero randomly.
    `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

    The **DropPath** function is defined as:

    .. math:: \text{DropPath}(x) = x * \text{Bernoulli}(p=1 - prob)

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    p : float, optional, default=0.2
        The dropping prob.
    increment : float, optional, default=0.
        The increment value to ``p``.
    training : bool, optional, default=True
        The training flag.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.
    slot : int, optional
        The optional slot index.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if not training:
        return input
    return _functions.DropPath \
        .instantiate(
            input.device,
            p=p,
            increment=increment,
            slot=slot,
        ).apply(input, inplace=inplace)


def elu(input, alpha=1., inplace=False):
    r"""Apply the exponential linear unit to input.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    The **ELU** function is defined as:

    .. math::
        \text{ELU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                \alpha * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    See Also
    --------
    `torch.nn.ELU`_ - Apply the exponential linear unit.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    alpha : float, optional, default=1.
        The value to :math:`\alpha`.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Elu \
        .instantiate(
            input.device,
            alpha=alpha,
        ).apply(input, inplace=inplace)


def group_norm(input, weight, bias, groups=32, eps=1e-5):
    r"""Apply the group normalization to input.
    `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    The normalization is defined as:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor
        The bias tensor.
    groups : int, optional, default=32
        The number of groups to split.
    eps : float, optional, default=1e-5
        The epsilon value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.GroupNorm \
        .instantiate(
            input.device,
            group=groups,
            eps=eps,
        ).apply(input, weight, bias)


def interpolate(
    input,
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=False,
):
    """Resize input via interpolating neighborhoods.

    Specify either ``size`` or ``scale_factor`` to compute output size:

    ```python
    x = torch.ones((1, 2, 3, 4))
    y = F.interpolate(x, size=6)  # Shape: (1, 2, 6, 6)
    z = F.interpolate(x, scale_factor=2)  # Shape: (1, 2, 6, 8)
    ```

    Set ``align_corners`` to determine the input coordinates in linear ``mode``:

    ```python
    # align_corners = False
    # Use half-pixel transformation
    scale = float(in_size) / float(out_size)
    in_coord = (out_coord + 0.5) * scale - 0.5

    # align_corners = True
    # Use align-corners transformation
    scale = float(in_size - 1) / float(out_size - 1)
    in_coord = out_coord * scale
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    size : Union[int, Sequence[int]], optional
        The output size.
    scale_factor : Union[number, Sequence[number]], optional
        The scale factor along each input dimension.
    mode : {'nearest', 'linear'}, optional
        The interpolation mode.
    align_corners : bool, optional, default=False
        Whether to align corners in linear interpolating.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if size is not None:
        size = nest.flatten(size)
    if scale_factor is not None:
        scale_factor = nest.flatten(scale_factor)
    return _functions.Resize \
        .instantiate(
            input.device,
            mode=mode.upper(),
            align_corners=align_corners,
            num_sizes=len(size) if size is not None else 0,
            num_scales=len(scale_factor) if scale_factor is not None else 0,
        ).apply(input, size, scale_factor)


def l1_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction='mean',
):
    r"""Compute the element-wise absolute value difference.

    The ``L1Loss`` function is defined as:

    .. math:: \text{L1Loss}(x, y) = |x - y|

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    target : dragon.vm.torch.Tensor
        The target tensor.
    size_average : bool, optional
        Whether to average the loss.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'none', 'mean', 'sum'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return _functions.L1Loss \
        .instantiate(
            input.device,
            reduction=reduction,
        ).apply([input, target])


def leaky_relu(input, negative_slope=0.01, inplace=False):
    r"""Apply the leaky rectified linear unit to input.

    The **LeakyReLU** function is defined as:

    .. math::
        \text{LeakyReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                slope * x, & \text{ otherwise }
            \end{cases}

    See Also
    --------
    `torch.nn.LeakyReLU`_ - Apply the leaky rectified linear unit.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    negative_slope : float, optional, default=0.01
        The slope to the negative side.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Relu \
        .instantiate(
            input.device,
            alpha=float(negative_slope),
        ).apply(input, inplace=inplace)


def linear(input, weight, bias=None):
    r"""Apply the linear transformation to input.

    .. math:: y = Wx + b

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The optional bias.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Linear \
        .instantiate(
            input.device,
        ).apply(input, weight, bias)


def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.):
    r"""Apply the local response normalization to input.
    `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

    The normalization is defined as:

    .. math::
        y_{i} = x_{i}\left(k + \frac{\alpha}{n}
            \sum_{j=\max(0, i-n/2)}^{\min(N-1,i+n/2)}x_{j}^2
        \right)^{-\beta}

    See Also
    --------
    `torch.nn.LocalResponseNorm`_ - Apply the local response normalization.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input.
    size : int, required
        The number of neighbouring channels to sum over.
    alpha : float, optional, default=0.0001
        The scale value :math:`\alpha`.
    beta : float, optional, default=0.75
        The exponent value :math:`\beta`.
    k : float, optional, default=1.
        The bias constant :math:`k`.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.LocalResponseNorm \
        .instantiate(
            input.device,
            size=size,
            alpha=float(alpha),
            beta=float(beta),
            bias=float(k),
        ).apply(input)


def log_softmax(input, dim):
    r"""Apply the composite of logarithm and softmax to input.

    The **LogSoftmax** function is defined as:

    .. math:: \text{LogSoftmax}(x_{i}) = \log(\frac{\exp(x_{i})}{\sum \exp(x_{j})})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input.
    dim : int
        The dimension to reduce.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return input - input.logsumexp(dim, keepdim=True)


def lstm_cell(input, cx):
    """Apply lstm cell to the input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input.
    cx : dragon.vm.torch.Tensor
        The previous cell state.

    Returns
    -------
    sequence of dragon.vm.torch.Tensor
        The **h** and **c**.

    """
    return _functions.LSTMCell \
        .instantiate(
            input.device,
        ).apply(input, cx)


def max_pool2d(
    input,
    kernel_size,
    stride=1,
    padding=0,
    ceil_mode=False,
    global_pooling=False,
):
    r"""Apply the 2d max pooling to input.

    The spatial output dimension is computed as:

    .. math::
        \text{Dim}_{out} = (\text{Dim}_{in} +
            2 * pad - \text{K}_{size}) / stride + 1

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kernel_size : Union[int, Sequence[int]]
        The size of sliding window.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of sliding window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero-padding size.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.
    global_pooling : bool, optional
        **True** to pool globally regardless of ``kernel_size``.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _pool(
        _pool_mode='MAX',
        _nd_util=utils._pair,
        _pool_fn=_functions.Pool2d,
        **locals()
    )


def mse_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction='mean',
):
    r"""Compute the element-wise squared error.

    The ``MSELoss`` function is defined as:

    .. math:: \text{MSELoss}(x, y) = (x - y)^{2}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    target : dragon.vm.torch.Tensor
        The target tensor.
    size_average : bool, optional
        Whether to average the loss.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'none', 'mean', 'sum'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return _functions.L2Loss \
        .instantiate(
            input.device,
            reduction=reduction,
        ).apply([input, target])


def nll_loss(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=None,
    reduce=None,
    reduction='mean',
):
    r"""Compute the negative likelihood loss with sparse labels.

    The **NLLLoss** function is defined as:

    .. math:: \text{NLLLoss}(p_{t}) = -\log(p_{t})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    target : dragon.vm.torch.Tensor
        The target tensor.
    weight : dragon.vm.torch.Tensor, optional
        The weight for each class.
    size_average : bool, optional
        Whether to average the loss.
    ignore_index : int, optional
        The label index to ignore.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'none', 'mean', 'sum', 'valid'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The loss.

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return _functions.NLLLoss \
        .instantiate(
            input.device,
            reduction=reduction,
            ignore_index=ignore_index,
        ).apply([input, target])


def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    r"""Apply the :math:`L_{p}` normalization to the input.

    The :math:`L_{p}` normalization is defined as:

    .. math:: v = \frac{v}{\left\|v\right\|_{p} + \epsilon}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    p : int, optional, default=2
        The exponent of norm.
    dim : int, optional, default=1
        The dimension to reduce.
    eps : float, optional, default=1e-12
        The value to :math:`\epsilon`.
    out : dragon.vm.torch.Tensor, optional
        The optional output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.LpNormalize \
        .instantiate(
            input.device,
            p=p,
            axis=dim,
            eps=eps,
        ).apply(input, out)


def pad(input, pad, mode='constant', value=0):
    r"""Pad the input according to the given sizes.

    The padded dimension is computed as:

    .. math:: \text{Dim}_{out} = \text{Dim}_{in} + pad_l + pad_r

    The ``pad`` should be a sequence of :math:`(N, 2)` values,
    where :math:`N` is the last n-dimensions to pad.

    Parameters
    ----------
    input :  dragon.vm.torch.Tensor
        The input tensor.
    pad : Sequence[int]
        The n-d padding sizes.
    mode : {'constant', 'reflect', 'replicate', 'circular'}, optional
        The padding mode.
    value : number, optional, default=0
        The value used in **constant** mode.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    ndim = input.ndimension()
    pads_begin, pads_end = [0] * ndim, [0] * ndim
    for i in range(len(pad) // 2):
        pads_begin[ndim - 1 - i] = pad[i * 2]
        pads_end[ndim - 1 - i] = pad[i * 2 + 1]
    return _functions.Pad \
        .instantiate(
            input.device,
            ndim=ndim,
            value=float(value),
            mode={'constant': 'CONSTANT',
                  'reflect': 'REFLECT',
                  'replicate': 'EDGE',
                  'circular': 'EDGE'}[mode],
        ).apply(input, pads_begin + pads_end)


def prelu(input, weight):
    r"""Apply parametric rectified linear unit to input.
    `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

    The **PReLU** function is defined as:

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
            x, & \text{ if } x \geq 0 \\
            weight * x, & \text{ otherwise }
        \end{cases}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.PRelu \
        .instantiate(
            input.device,
        ).apply(input, weight)


def relu(input, inplace=False):
    r"""Apply the rectified linear unit to input.
    `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    The **ReLU** function is defined as:

    .. math::
        \text{ReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                0, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return leaky_relu(input, 0., inplace=inplace)


def relu6(input, inplace=False):
    r"""Apply the clipped-6 rectified linear unit to input.
    `[Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_.

    The **ReLU-6** function is defined as:

    .. math::
        \text{ReLU-6}(x) =
            \begin{cases}
                \min(x, 6), & \text{ if } x \geq 0 \\
                0, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Relu6 \
        .instantiate(
            input.device,
        ).apply(input, inplace=inplace)


def selu(input, inplace=False):
    r"""Apply the scaled exponential linear unit to input.
    `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    The **SELU** function is defined as:

    .. math::
        \text{SELU}(x) = 1.0507 *
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                1.67326 * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _activation(input, inplace, 'Selu')


def sigmoid(input, inplace=False):
    r"""Apply the sigmoid function to input.

    The **Sigmoid** function is defined as:

    .. math:: \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _activation(input, inplace, 'Sigmoid')


def sigmoid_focal_loss(
    input,
    target,
    alpha=0.25,
    gamma=2.,
    weight=None,
    size_average=None,
    negative_index=None,
    reduce=None,
    reduction='valid',
):
    r"""Compute the sigmoid focal loss with sparse labels.
    `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`__.

    The **FocalLoss** function is defined as:

    .. math:: \text{FocalLoss}(p_{t}) = -(1 - p_{t})^{\gamma}\log(p_{t})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    target : dragon.vm.torch.Tensor
        The target tensor.
    alpha : float, optional, default=0.25
        The scale factor on the rare class.
    gamma : float, optional, default=2.
        The exponential decay factor on the easy examples.
    weight : dragon.vm.torch.Tensor, optional
        The weight for each class.
    size_average : bool, optional
        Whether to average the loss.
    negative_index : int, optional
        The negative label index.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'none', 'mean', 'sum', 'valid'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return _functions.SigmoidFocalLoss \
        .instantiate(
            input.device,
            alpha=float(alpha),
            gamma=float(gamma),
            reduction=reduction,
            negative_index=negative_index,
        ).apply([input, target])


def smooth_l1_loss(
    input,
    target,
    beta=1.,
    size_average=None,
    reduce=None,
    reduction='mean',
):
    r"""Compute the element-wise error transited from L1 and L2.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    The **SmoothL1Loss** function is defined as:

    .. math::
        \text{SmoothL1Loss}(x, y) =
            \begin{cases}
                0.5 * (x - y)^{2} / beta, & \text{ if } |x - y| < beta \\
                |x - y| - 0.5 * beta, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    target : dragon.vm.torch.Tensor
        The target tensor.
    beta : float, optional, default=1.
        The transition point from L1 to L2 loss.
    size_average : bool, optional
        Whether to average the loss.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'batch_size', 'sum', mean'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return _functions.SmoothL1Loss \
        .instantiate(
            input.device,
            beta=float(beta),
            reduction=reduction,
        ).apply([input, target])


def softmax(input, dim, inplace=False):
    r"""Apply the softmax function to input.

    The **Softmax** function is defined as:

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_{i})}{\sum_{j} \exp(x_{j})}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    dim : int
        The dimension to reduce.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _functions.Softmax \
        .instantiate(
            input.device,
            axis=dim,
        ).apply(input, inplace=inplace)


def sync_batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training=False,
    momentum=0.1,
    eps=1e-5,
    process_group=None,
):
    r"""Apply the sync batch normalization over input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The moving average of stats are calculated as:

    .. math::
        x_{moving} \leftarrow (1 - momentum) * x_{moving} + momentum * x_{stat}

    Additionally, you can specify ``process_group`` to perform synchronization.

    If not, value returning from ``dragon.distributed.get_group(...)`` will be used.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    running_mean : dragon.vm.torch.Tensor
        The running mean.
    running_var : dragon.vm.torch.Tensor
        The running var.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor
        The bias tensor.
    training : bool, optional, default=False
        The flag to determine the stats.
    momentum : float, optional, default=0.1
        The momentum to the moving average.
    eps : float, optional, default=1e-5
        The epsilon value.
    process_group : dragon.ProcessGroup, optional
        The group for communication.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if process_group is None:
        raise ValueError('<process_group> is required.')
    return _functions.SyncBatchNorm \
        .instantiate(
            input.device,
            training=training,
            momentum=momentum,
            eps=eps,
            process_group=process_group,
        ).apply(input, running_mean, running_var, weight, bias)


def tanh(input, inplace=False):
    r"""Apply the tanh function to input.

    The **Tanh** function is defined as:

    .. math:: \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return _activation(input, inplace, 'Tanh')


def upsample(
    input,
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=False,
):
    """Upsample input via interpolating neighborhoods.

    Specify either ``size`` or ``scale_factor`` to compute output size:

    ```python
    x = torch.ones((1, 2, 3, 4))
    y = F.interpolate(x, size=6)  # Shape: (1, 2, 6, 6)
    z = F.interpolate(x, scale_factor=2)  # Shape: (1, 2, 6, 8)
    ```

    Set ``align_corners`` to determine the input coordinates in linear ``mode``:

    ```python
    # align_corners = False
    # Use half-pixel transformation
    scale = float(in_size) / float(out_size)
    in_coord = (out_coord + 0.5) * scale - 0.5

    # align_corners = True
    # Use align-corners transformation
    scale = float(in_size - 1) / float(out_size - 1)
    in_coord = out_coord * scale
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    size : Union[int, Sequence[int]], optional
        The output size.
    scale_factor : Union[number, Sequence[number]], optional
        The scale factor along each input dimension.
    mode : {'nearest', 'linear'}, optional
        The interpolation mode.
    align_corners : bool, optional, default=False
        Whether to align corners in linear interpolating.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return interpolate(input, size, scale_factor, mode, align_corners)


def upsample_bilinear(input, size=None, scale_factor=None):
    """Upsample input via bilinear interpolating.

    Examples:

    ```python
    x = torch.ones((1, 2, 3, 4))
    y = F.upsample_bilinear(x, size=6)  # Shape: (1, 2, 6, 6)
    z = F.upsample_bilinear(x, scale_factor=2)  # Shape: (1, 2, 6, 8)
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    size : Union[int, Sequence[int]], optional
        The output size.
    scale_factor : Union[number, Sequence[number]], optional
        The scale factor along each input dimension.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return interpolate(input, size, scale_factor, 'linear', align_corners=True)


def upsample_nearest(input, size=None, scale_factor=None):
    """Upsample input via nearest interpolating.

    Examples:

    ```python
    x = torch.ones((1, 2, 3, 4))
    y = F.upsample_nearest(x, size=6)  # Shape: (1, 2, 6, 6)
    z = F.upsample_nearest(x, scale_factor=2)  # Shape: (1, 2, 6, 8)
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    size : Union[int, Sequence[int]], optional
        The output size.
    scale_factor : Union[number, Sequence[number]], optional
        The scale factor along each input dimension.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return interpolate(input, size, scale_factor, 'nearest')


def _activation(input, inplace=False, _op_type=''):
    return _functions._Activation \
        .instantiate(
            input.device,
            op_type=_op_type,
        ).apply(input, inplace=inplace)


def _conv(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=None,
    _nd_util=utils._pair,
    _conv_fn=_functions.Conv2d,
):
    weight_shape = list(weight.shape)
    kernel_shape = weight_shape[2:]
    return _conv_fn.instantiate(
        input.device,
        in_channels=weight_shape[1],
        out_channels=weight_shape[0],
        kernel_shape=kernel_shape,
        strides=_nd_util(stride),
        pads=_nd_util(padding),
        dilations=_nd_util(dilation),
        group=groups,
        bias=bias is not None,
        dtype=weight.dtype,
    ).apply(input, weight, bias)


def _conv_transpose(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
    _nd_util=utils._pair,
    _conv_fn=_functions.ConvTranspose2d,
):
    weight_shape = list(weight.shape)
    kernel_shape = weight_shape[2:]
    return _conv_fn.instantiate(
        input.device,
        in_channels=weight_shape[0],
        out_channels=weight_shape[1],
        kernel_shape=kernel_shape,
        strides=_nd_util(stride),
        pads=_nd_util(padding),
        dilations=_nd_util(dilation),
        group=groups,
        output_padding=_nd_util(output_padding),
        bias=bias is not None,
        dtype=weight.dtype,
    ).apply(input, weight, bias)


def _pool(
    input,
    kernel_size,
    stride=1,
    padding=0,
    ceil_mode=False,
    global_pooling=False,
    _pool_mode='MAX',
    _nd_util=utils._pair,
    _pool_fn=_functions.Pool2d,
):
    return _pool_fn.instantiate(
        input.device,
        kernel_shape=_nd_util(kernel_size),
        strides=_nd_util(stride),
        pads=_nd_util(padding),
        mode=_pool_mode,
        ceil_mode=ceil_mode,
        global_pooling=global_pooling,
    ).apply(input)
