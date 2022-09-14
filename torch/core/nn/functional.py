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
from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.nn import _reduction
from dragon.vm.torch.core.nn.modules import utils


def adaptive_avg_pool1d(input, output_size):
    """Apply the 1d adaptive average pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    output_size : Union[int, Sequence[int]]
        The target output size.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.AdaptiveAvgPool1d(...)`_

    """
    args = utils._get_adaptive_pool_args(
        input.size()[-1:], utils._single(output_size))
    return _pool('AVG', utils._single, input, **args)


def adaptive_avg_pool2d(input, output_size):
    """Apply the 2d adaptive average pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    output_size : Union[int, Sequence[int]]
        The target output size.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.AdaptiveAvgPool2d(...)`_

    """
    args = utils._get_adaptive_pool_args(
        input.size()[-2:], utils._pair(output_size))
    return _pool('AVG', utils._pair, input, **args)


def adaptive_avg_pool3d(input, output_size):
    """Apply the 3d adaptive average pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    output_size : Union[int, Sequence[int]]
        The target output size.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.AdaptiveAvgPool3d(...)`_

    """
    args = utils._get_adaptive_pool_args(
        input.size()[-3:], utils._triple(output_size))
    return _pool('AVG', utils._triple, input, **args)


def adaptive_max_pool1d(input, output_size):
    """Apply the 1d adaptive max pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    output_size : Union[int, Sequence[int]]
        The target output size.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.AdaptiveMaxPool1d(...)`_

    """
    args = utils._get_adaptive_pool_args(
        input.size()[-1:], utils._single(output_size))
    return _pool('MAX', utils._single, input, **args)


def adaptive_max_pool2d(input, output_size):
    """Apply the 2d adaptive max pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    output_size : Union[int, Sequence[int]]
        The target output size.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.AdaptiveMaxPool2d(...)`_

    """
    args = utils._get_adaptive_pool_args(
        input.size()[-2:], utils._pair(output_size))
    return _pool('MAX', utils._pair, input, **args)


def adaptive_max_pool3d(input, output_size):
    """Apply the 3d adaptive max pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    output_size : Union[int, Sequence[int]]
        The target output size.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.AdaptiveMaxPool3d(...)`_

    """
    args = utils._get_adaptive_pool_args(
        input.size()[-3:], utils._triple(output_size))
    return _pool('MAX', utils._triple, input, **args)


def affine(input, weight, bias=None, dim=-1, out=None):
    r"""Apply affine transformation to input.

    .. math:: \text{out} = \text{input} \times \text{weight} + \text{bias}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.
    dim : Union[int, Sequence[int]], optional
        The dimension to apply.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.Affine(...)`_

    """
    return Function.apply(
        'Affine', input.device,
        [input, weight] + ([bias] if bias else []), outputs=[out],
        axes=nest.flatten(dim))


def avg_pool1d(input, kernel_size, stride=1, padding=0, ceil_mode=False):
    """Apply the 1d average pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kernel_size : Union[int, Sequence[int]]
        The size of pooling window.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of pooling window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.AvgPool1d(...)`_

    """
    return _pool('AVG', utils._single, **locals())


def avg_pool2d(input, kernel_size, stride=1, padding=0, ceil_mode=False):
    """Apply the 2d average pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kernel_size : Union[int, Sequence[int]]
        The size of pooling window.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of pooling window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.AvgPool2d(...)`_

    """
    return _pool('AVG', utils._pair, **locals())


def avg_pool3d(input, kernel_size, stride=1, padding=0, ceil_mode=False):
    """Apply the 3d average pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kernel_size : Union[int, Sequence[int]]
        The size of pooling window.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of pooling window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.AvgPool3d(...)`_

    """
    return _pool('AVG', utils._triple, **locals())


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
    """Apply the batch normalization to input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

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

    See Also
    --------
    `torch.nn.BatchNorm2d(...)`_

    """
    return Function.apply(
        'BatchNorm', input.device,
        [input, weight, bias, running_mean, running_var],
        axis=1, epsilon=eps, use_stats=int(not training),
        momentum=1.0 - momentum)


def binary_cross_entropy_with_logits(
    input,
    target,
    weight=None,
    size_average=None,
    reduce=None,
    reduction='mean',
    pos_weight=None,
):
    """Compute the sigmoid cross entropy with contiguous target.

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

    See Also
    --------
    `torch.nn.BCEWithLogitsLoss(...)`_

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return Function.apply(
        'SigmoidCrossEntropyLoss', input.device,
        [input, target], reduction=reduction.upper())


def channel_norm(input, mean, std, dim=-1, dtype='float32', dims=None):
    """Apply the normalization to each channel of input.

    :attr:`dim` can be negative:

    ```python
    m = s = (1., 1., 1.)
    x = torch.tensor([1, 2, 3])
    print(nn.functional.channel_norm(x, m, s, dim=0))  # [0., 1., 2.]
    print(nn.functional.channel_norm(x, m, s, dim=-1))  # Equivalent
    ```

    If :attr:`dims` provided, :attr:`dim` is selected from the output layout:

    ```python
    m, s = (1., 2., 3.), (1., 1., 1.)
    x = torch.tensor([[1, 2, 3]])
    # Provided 3 values to normalize the last dimension
    # with length 1, only the first value will be taken
    print(nn.functional.channel_norm(x, m, s, dims=(1, 0)))  # [[0.], [1.], [2.]]
    ```

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    mean : Sequence[float], required
        The mean to subtract.
    std : Sequence[float], required
        The standard deviation to divide.
    dim : int, optional, default=-1
        The channel dimension.
    dtype : str, optional, default='float32'
        The output data type.
    dims : Sequence[int], optional
        The order of output dimensions.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'ChannelNorm', input.device, [input],
        axis=dim, mean=mean, std=std, dtype=dtype,
        ndim=len(dims) if dims is not None else 0, perm=dims)


def channel_shuffle(input, groups):
    """Apply group shuffle to each channel of input.
    `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    groups : int
        The number of shuffle groups.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.ChannelShuffle(...)`_

    """
    return Function.apply(
        'ChannelShuffle', input.device, [input], axis=1, group=groups)


def conv1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """Apply the 1d convolution to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilation : Union[int, Sequence[int]], optional, default=1
        The rate of dilated kernel.
    groups : int, optional, default=1
        The number of groups to split input channels.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.Conv1d(...)`_

    """
    return _conv('Conv', utils._single, **locals())


def conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """Apply the 2d convolution to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilation : Union[int, Sequence[int]], optional, default=1
        The rate of dilated kernel.
    groups : int, optional, default=1
        The number of groups to split input channels.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.Conv2d(...)`_

    """
    return _conv('Conv', utils._pair, **locals())


def conv3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """Apply the 3d convolution to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilation : Union[int, Sequence[int]], optional, default=1
        The rate of dilated kernel.
    groups : int, optional, default=1
        The number of groups to split input channels.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.Conv3d(...)`_

    """
    return _conv('Conv', utils._triple, **locals())


def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    """Apply the 1d deconvolution to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    output_padding : int, optional, default=1
        The additional size added to the output shape.
    groups : int, optional, default=1
        The number of groups to split input channels.
    dilation : Union[int, Sequence[int]], optional, default=1
        The rate of dilated kernel.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.ConvTranspose1d(...)`_

    """
    return _conv_transpose('ConvTranspose', utils._single, **locals())


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
    """Apply the 2d deconvolution to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    output_padding : int, optional, default=1
        The additional size added to the output shape.
    groups : int, optional, default=1
        The number of groups to split input channels.
    dilation : Union[int, Sequence[int]], optional, default=1
        The rate of dilated kernel.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.ConvTranspose2d(...)`_

    """
    return _conv_transpose('ConvTranspose', utils._pair, **locals())


def conv_transpose3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    """Apply the 3d deconvolution to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    output_padding : int, optional, default=1
        The additional size added to the output shape.
    groups : int, optional, default=1
        The number of groups to split input channels.
    dilation : Union[int, Sequence[int]], optional, default=1
        The rate of dilated kernel.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.ConvTranspose3d(...)`_

    """
    return _conv_transpose('ConvTranspose', utils._triple, **locals())


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Compute the cosine similarity between inputs.

    Parameters
    ----------
    x1 : dragon.vm.torch.Tensor
        The first input tensor.
    x2 : dragon.vm.torch.Tensor
        The second input tensor.
    dim : int, optional, default=1
        The vector dimension.
    eps : float, optional, default=1e-8
        The epsilon value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.CosineSimilarity(...)`_

    """
    x1 = normalize(x1, p=2, dim=dim, eps=eps)
    x2 = normalize(x2, p=2, dim=dim, eps=eps)
    return (x1 * x2).sum(dim)


def cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=None,
    reduce=None,
    reduction='mean',
):
    """Compute the softmax cross entropy with sparse target.

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
        The ignored value of target.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'none', 'mean', 'sum', 'valid'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The loss.

    See Also
    --------
    `torch.nn.CrossEntropyLoss(...)`_

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return Function.apply(
        'SoftmaxCrossEntropyLoss', input.device, [input, target],
        axis=1, ignore_index=ignore_index, reduction=reduction.upper())


def ctc_loss(input, target, padding_mask=-1, reduction='mean'):
    """Compute the ctc loss.
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

    See Also
    --------
    `torch.nn.CTCLoss(...)`_

    """
    return Function.apply(
        'CTCLoss', input.device, [input, target],
        padding_mask=padding_mask, reduction=reduction.upper())


def depthwise_conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
):
    """Apply the 2d depthwise convolution to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.
    stride : Sequence[int], default=1
        The stride of convolution window.
    padding : Sequence[int], default=0
        The zero padding size.
    dilation : Sequence[int], default=1
        The rate of dilated kernel.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.DepthwiseConv2d(...)`_

    """
    return _conv('DepthwiseConv', utils._pair, **locals())


def dropout(input, p=0.5, training=True, inplace=False):
    """Set the elements of the input to zero randomly.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    p : float, optional, default=0.5
        The probability to zero an element.
    training : bool, optional, default=True
        Apply dropping if ``True``.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.Dropout(...)`_

    """
    if not training or p <= 0:
        return input
    return Function.apply(
        'Dropout', input.device, [input],
        outputs=[input if inplace else None], ratio=p)


def drop_block2d(input, p=0.5, block_size=1, training=True, inplace=False):
    """Set the blocks over input to zero randomly.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    p : float, optional, default=0.5
        The probability to zero an element.
    block_size : int, optional, default=1
        The spatial block size.
    training : bool, optional, default=True
        Apply dropping if ``True``.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.DropBlock2d(...)`_

    """
    if not training or p <= 0:
        return input
    return Function.apply(
        'DropBlock', input.device, [input],
        outputs=[input if inplace else None], block_size=block_size, ratio=p)


def drop_path(input, p=0.2, training=True, inplace=False):
    r"""Set the examples over input to zero randomly.
    `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    p : float, optional, default=0.2
        The probability to zero an element.
    training : bool, optional, default=True
        Apply dropping if ``True``.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.DropPath(...)`_

    """
    if not training or p <= 0:
        return input
    return Function.apply(
        'DropPath', input.device, [input],
        outputs=[input if inplace else None], ratio=p)


def elu(input, alpha=1.0, inplace=False):
    r"""Apply the exponential linear unit to input.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    alpha : float, optional, default=1.0
        The value to :math:`\alpha`.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.ELU(...)`_

    """
    return Function.apply(
        'Elu', input.device, [input],
        outputs=[input if inplace else None], alpha=float(alpha))


def embedding(input, weight, padding_idx=None):
    """Lookup the embedding matrix using index.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The index tensor.
    weight : dragon.vm.torch.Tensor
        The embedding matrix.
    padding_idx : int, optional
        The position where to return zeros.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    if padding_idx is not None:
        num_embeddings = weight.size(0)
        if padding_idx > 0:
            if padding_idx >= num_embeddings:
                raise ValueError('<padding_idx> must be within embedding matrix.')
        elif padding_idx < 0:
            if padding_idx < -num_embeddings:
                raise ValueError('<padding_idx> must be within embedding matrix.')
            padding_idx = num_embeddings + padding_idx
        weight[padding_idx] = 0
    return weight.index_select(0, input)


def gelu(input, approximate='none'):
    r"""Apply the gaussian error linear unit to input.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    The **GELU** function is defined as:

    .. math:: \text{GELU}(x) = x\cdot\frac{1}{2}[1 + \text{erf}(x / \sqrt{2})]

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    approximate : str, optional, default='none'
        The approximate algorithm.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.GELU(...)`_

    """
    approximate = approximate == 'tanh'
    return Function.apply('Gelu', input.device, [input], approximate=approximate)


def group_norm(input, num_groups, weight, bias, eps=1e-5):
    """Apply the group normalization to input.
    `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    num_groups : int
        The number of groups.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor
        The bias tensor.
    eps : float, optional, default=1e-5
        The epsilon value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.GroupNorm(...)`_

    """
    return Function.apply(
        'GroupNorm', input.device, [input, weight, bias],
        axis=1, group=num_groups, epsilon=eps)


def hardsigmoid(input, inplace=False):
    r"""Apply the hard sigmoid function to input.

    The **HardSigmoid** function is defined as:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
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

    See Also
    --------
    `torch.nn.Hardsigmoid(...)`_

    """
    return Function.apply(
        'HardSigmoid', input.device, [input],
        outputs=[input if inplace else None], alpha=1. / 6., beta=0.5)


def hardswish(input):
    r"""Apply the hard swish function to input.
    `[Howard et.al, 2019] <https://arxiv.org/abs/1905.02244>`_.

    The **HardSwish** function is defined as:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.Hardswish(...)`_

    """
    return Function.apply('HardSwish', input.device, [input])


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

    See Also
    --------
    `torch.nn.Upsample(...)`_

    """
    if size is not None:
        size = nest.flatten(size)
    if scale_factor is not None:
        scale_factor = nest.flatten(scale_factor)
    mode = mode.upper()
    mode = mode.replace('BILINEAR', 'LINEAR')
    mode = mode.replace('TRILINEAR', 'LINEAR')
    return Function.apply(
        'Resize', input.device, [input],
        mode=mode, align_corners=align_corners,
        num_sizes=len(size) if size is not None else 0,
        num_scales=len(scale_factor) if scale_factor is not None else 0,
        sizes=size, scales=scale_factor)


def kl_div(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction='mean',
    log_target=False,
):
    """Compute the Kullback-Leibler divergence.

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
    reduction : {'none', 'batchmean', 'mean', 'sum'}, optional
        The reduce method.
    log_target : bool, optional, default=False
        The flag indicating whether ``target`` is passed in log space.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.KLDivLoss(...)`_

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    if not log_target:
        out = target * (target.log() - input)
    else:
        out = target.exp() * (target - input)
    if reduction == 'none':
        return out
    elif reduction == 'batchmean':
        return out.sum() / input.size()[0]
    elif reduction == 'mean':
        return out.mean()
    else:
        return out.sum()


def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """Compute the element-wise absolute value difference.

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

    See Also
    --------
    `torch.nn.L1Loss(...)`_

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return Function.apply(
        'L1Loss', input.device, [input, target],
        reduction=reduction.upper())


def layer_norm(input, normalized_shape, weight, bias, eps=1e-5):
    """Apply the layer normalization to input.
    `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    normalized_shape : Sequence[int]
        The size normalized over the last dimensions.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor
        The bias tensor.
    eps : float, optional, default=1e-5
        The epsilon value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.LayerNorm(...)`_

    """
    return Function.apply(
        'LayerNorm', input.device, [input, weight, bias],
        axis=input.ndimension() - len(normalized_shape), epsilon=eps)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    r"""Apply the leaky rectified linear unit to input.

    The **LeakyReLU** function is defined as:

    .. math::
        \text{LeakyReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                slope * x, & \text{ otherwise }
            \end{cases}

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

    See Also
    --------
    `torch.nn.LeakyReLU(...)`_

    """
    return Function.apply(
        'Relu', input.device, [input],
        outputs=[input if inplace else None], alpha=float(negative_slope))


def linear(input, weight, bias=None):
    r"""Apply the linear transformation to input.

    .. math:: \text{out} = \text{input} \times \text{weight}^{T} + \text{bias}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    weight : dragon.vm.torch.Tensor
        The weight tensor.
    bias : dragon.vm.torch.Tensor, optional
        The bias tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.Linear(...)`_

    """
    return Function.apply(
        'Gemm', input.device,
        [input, weight] + ([bias] if bias else []),
        transA=False, transB=True)


def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.):
    r"""Apply the local response normalization to input.
    `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

    The normalization is defined as:

    .. math::
        y_{i} = x_{i}\left(k + \frac{\alpha}{n}
            \sum_{j=\max(0, i-n/2)}^{\min(N-1,i+n/2)}x_{j}^2
        \right)^{-\beta}

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

    See Also
    --------
    `torch.nn.LocalResponseNorm(...)`_

    """
    return Function.apply(
        'LRN', input.device, [input],
        size=size, alpha=float(alpha), beta=float(beta), bias=float(k))


def log_softmax(input, dim, inplace=False):
    r"""Apply the composite of logarithm and softmax to input.

    The **LogSoftmax** function is defined as:

    .. math:: \text{LogSoftmax}(x_{i}) = \log(\frac{\exp(x_{i})}{\sum \exp(x_{j})})

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input.
    dim : int
        The dimension to reduce.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.LogSoftmax(...)`_

    """
    return Function.apply(
        'LogSoftmax', input.device, [input],
        outputs=[input if inplace else None], axis=dim)


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

    See Also
    --------
    `torch.nn.LSTMCell(...)`_

    """
    return Function.apply('LSTMCell', input.device, [input, cx])


def max_pool1d(input, kernel_size, stride=1, padding=0, ceil_mode=False):
    """Apply the 1d max pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kernel_size : Union[int, Sequence[int]]
        The size of pooling window.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of pooling window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.MaxPool1d(...)`_

    """
    return _pool('MAX', utils._single, **locals())


def max_pool2d(input, kernel_size, stride=1, padding=0, ceil_mode=False):
    """Apply the 2d max pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kernel_size : Union[int, Sequence[int]]
        The size of pooling window.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of pooling window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.MaxPool2d(...)`_

    """
    return _pool('MAX', utils._pair, **locals())


def max_pool3d(input, kernel_size, stride=1, padding=0, ceil_mode=False):
    """Apply the 3d max pooling to input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kernel_size : Union[int, Sequence[int]]
        The size of pooling window.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of pooling window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.MaxPool3d(...)`_

    """
    return _pool('MAX', utils._triple, **locals())


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """Compute the element-wise squared error.

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

    See Also
    --------
    `torch.nn.MSELoss(...)`_

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return Function.apply(
        'L2Loss', input.device, [input, target],
        reduction=reduction.upper())


def multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    dropout_p=0.,
    training=True,
    need_weights=True,
    key_padding_mask=None,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
):
    """Apply the multihead attention to input.
    `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

    Parameters
    ----------
    query : dragon.vm.torch.Tensor
        The query tensor.
    key : dragon.vm.torch.Tensor
        The key tensor.
    value : dragon.vm.torch.Tensor
        The value tensor.
    embed_dim_to_check : int
        The dimension of input embeddings.
    num_heads : int
        The number of parallel heads.
    in_proj_weight : dragon.vm.torch.Tensor
        The weight tensor for input projection.
    in_proj_bias : dragon.vm.torch.Tensor
        The bias tensor for input projection.
    out_proj_weight: dragon.vm.torch.Tensor
        The weight tensor for output projection.
    out_proj_bias: dragon.vm.torch.Tensor
        The bias tensor for output projection.
    dropout_p: float, optional, default=0.
        The probability to set the attention to zero.
    training: bool, optional, default=True
        Apply dropout if ``True``.
    need_weights : bool, optional, default=True
        Return the attention weights or not.
    key_padding_mask: dragon.vm.torch.Tensor, optional
        The mask to prevents attention to padded keys.
    attn_mask: dragon.vm.torch.Tensor, optional
        The mask to prevents attention to certain positions.
    use_separate_proj_weight : bool, optional, default=False
        Provide separate projection weights or not.
    q_proj_weight : dragon.vm.torch.Tensor, optional
        The separate weight tensor for query projection.
    k_proj_weight : dragon.vm.torch.Tensor, optional
        The separate weight tensor for key projection.
    v_proj_weight : dragon.vm.torch.Tensor, optional
        The separate weight tensor for value projection.

    Returns
    -------
    Tuple[dragon.vm.torch.Tensor, dragon.vm.torch.Tensor]
        The output and attention weights tensor.

    See Also
    --------
    `torch.nn.MultiheadAttention(...)`_

    """
    tgt_len, bsz, embed_dim = query.size()
    src_len = key.size(0)
    assert embed_dim == embed_dim_to_check
    assert src_len == value.size(0) and key.size(1) == value.size(1)
    head_dim = embed_dim // num_heads

    def to_qkv(input, weight, bias, num_proj=1):
        """Compute input projections via a single matmul."""
        if num_proj > 1:
            qkv_shape = (tgt_len, bsz, num_proj, num_heads, head_dim)
            qkv = linear(input, weight, bias).reshape_(qkv_shape)
            return qkv.permute(2, 1, 3, 0, 4)
        qkv_shape = (tgt_len, bsz, num_heads, head_dim)
        qkv = linear(input, weight, bias).reshape_(qkv_shape)
        return qkv.permute(1, 2, 0, 3)

    q, k, v = None, None, None
    if not use_separate_proj_weight:
        if (query is key) and (key is value):
            # Parallelism for self attention.
            qkv = to_qkv(query, in_proj_weight, in_proj_bias, 3)
            q, k, v = qkv.unbind(dim=0, copy=False)
        elif key is value:
            # Parallelism for encode-decoder attention.
            q_proj_weight = in_proj_weight[:embed_dim, :]
            kv_proj_weight = in_proj_weight[embed_dim:, :]
            q_proj_bias = kv_proj_bias = in_proj_bias
            if in_proj_bias is not None:
                q_proj_bias = in_proj_bias[:embed_dim]
                kv_proj_bias = in_proj_bias[embed_dim:]
            q = to_qkv(query, q_proj_weight, q_proj_bias)
            kv = to_qkv(key, kv_proj_weight, kv_proj_bias, 2)
            k, v = kv.unbind(dim=0, copy=False)
    if q is None:
        q_proj_bias = k_proj_bias = v_proj_bias = in_proj_bias
        if use_separate_proj_weight and q_proj_weight is None:
            q_proj_weight = in_proj_weight[:embed_dim, :]
            k_proj_weight = in_proj_weight[embed_dim:embed_dim * 2, :]
            v_proj_weight = in_proj_weight[embed_dim * 2:, :]
        if in_proj_bias is not None:
            q_proj_bias = in_proj_bias[:embed_dim]
            k_proj_bias = in_proj_bias[embed_dim:embed_dim * 2]
            v_proj_bias = in_proj_bias[embed_dim * 2:]
        q = to_qkv(query, q_proj_weight, q_proj_bias)
        k = to_qkv(key, k_proj_weight, k_proj_bias)
        v = to_qkv(value, v_proj_weight, v_proj_bias)
    attn = q @ (k.transpose(-2, -1).mul_(float(head_dim) ** -0.5))
    assert attn.size() == (bsz, num_heads, tgt_len, src_len)
    if attn_mask is not None:
        if attn_mask.dtype == 'bool' or attn_mask.dtype == 'uint8':
            attn.masked_fill_(attn_mask, float('-inf'))
        else:
            attn += attn_mask
    if key_padding_mask is not None:
        if key_padding_mask.size() != attn.size():
            key_padding_mask.reshape_((bsz, 1, 1, src_len))
        attn.masked_fill_(key_padding_mask, float('-inf'))
    attn = softmax(attn, dim=-1, inplace=True)
    attn = dropout(attn, p=dropout_p, training=training)
    output = (attn @ v).permute(2, 0, 1, 3)
    output = output.reshape_((tgt_len, bsz, embed_dim))
    output = linear(output, out_proj_weight, out_proj_bias)
    weights = attn.mean(dim=1) if need_weights else None
    return output, weights


def nll_loss(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=None,
    reduce=None,
    reduction='mean',
):
    """Compute the negative likelihood loss.

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
        The ignored value of target.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'none', 'mean', 'sum', 'valid'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The loss.

    See Also
    --------
    `torch.nn.NLLLoss(...)`_

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return Function.apply(
        'NLLLoss', input.device, [input, target],
        axis=1, ignore_index=ignore_index, reduction=reduction.upper())


def normalize(input, p=2, dim=1, end_dim=None, eps=1e-12, out=None):
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
        The first dimension to reduce.
    end_dim : int, optional
        The last dimension to reduce.
    eps : float, optional, default=1e-12
        The value to :math:`\epsilon`.
    out : dragon.vm.torch.Tensor, optional
        The output tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'LpNorm', input.device, [input], outputs=[out],
        p=p, axis=dim, end_axis=end_dim, epsilon=eps, reduction='SUM')


def one_hot(tensor, num_classes, on_value=1, off_value=0):
    """Return the one-hot representation of input.

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    num_classes : int
        The number of classes.
    on_value : number, optional, default=1
        The value for equal branch.
    off_value : number, optional, default=0
        The value for not-equal branch.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'OneHot', tensor.device, [tensor], depth=num_classes,
        on_value=float(on_value), off_value=float(off_value))


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

    See Also
    --------
    `torch.nn.ConstantPad2d(...)`_,
    `torch.nn.ReflectionPad2d(...)`_,
    `torch.nn.ReplicationPad2d(...)`_

    """
    ndim = input.ndimension()
    pads_begin, pads_end = [0] * ndim, [0] * ndim
    for i in range(len(pad) // 2):
        pads_begin[ndim - 1 - i] = pad[i * 2]
        pads_end[ndim - 1 - i] = pad[i * 2 + 1]
    mode = {'constant': 'CONSTANT', 'reflect': 'REFLECT',
            'replicate': 'EDGE', 'circular': 'EDGE'}[mode]
    return Function.apply(
        'Pad', input.device, [input], mode=mode, value=float(value),
        ndim=ndim, pads=pads_begin + pads_end)


def pixel_shuffle(input, upscale_factor, inplace=False):
    """Rearrange depth elements of input into pixels.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    upscale_factor : int
        The factor to upscale pixels.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'DepthToSpace', input.device, [input],
        outputs=[input if inplace else None],
        block_size=int(upscale_factor), mode='CRD', data_format='NCHW')


def pixel_unshuffle(input, downscale_factor, inplace=False):
    """Rearrange pixels of input into depth elements.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    downscale_factor : int
        The factor to downscale pixels.
    inplace : bool, optional, default=False
        Whether to do the operation in-place.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Function.apply(
        'SpaceToDepth', input.device, [input],
        outputs=[input if inplace else None],
        block_size=int(downscale_factor), mode='CRD', data_format='NCHW')


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

    See Also
    --------
    `torch.nn.PReLU(...)`_

    """
    return Function.apply('PRelu', input.device, [input, weight])


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

    See Also
    --------
    `torch.nn.ReLU(...)`_

    """
    return Function.apply(
        'Relu', input.device, [input],
        outputs=[input if inplace else None], alpha=0.)


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

    See Also
    --------
    `torch.nn.ReLU6(...)`_

    """
    return Function.apply(
        'Relu', input.device, [input],
        outputs=[input if inplace else None], alpha=0., max_value=6.)


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

    See Also
    --------
    `torch.nn.SELU(...)`_

    """
    return Function.apply(
        'Selu', input.device, [input],
        outputs=[input if inplace else None], alpha=1.67326, gamma=1.0507)


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

    See Also
    --------
    `torch.nn.Sigmoid(...)`_

    """
    return Function.apply(
        'Sigmoid', input.device, [input],
        outputs=[input if inplace else None])


def sigmoid_focal_loss(
    input,
    target,
    alpha=0.25,
    gamma=2.,
    weight=None,
    size_average=None,
    start_index=0,
    reduce=None,
    reduction='mean',
):
    """Compute the sigmoid focal loss.
    `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`__.

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
    start_index : int, optional, default=0
        The start value of index.
    reduce : bool, optional
        Whether to reduce the loss.
    reduction : {'none', 'mean', 'sum', 'valid'}, optional
        The reduce method.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.SigmoidFocalLoss(...)`_

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return Function.apply(
        'SigmoidFocalLoss', input.device, [input, target],
        axis=1, alpha=float(alpha), gamma=float(gamma),
        start_index=start_index, reduction=reduction.upper())


def silu(input):
    r"""Apply the sigmoid linear unit to input.
    `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

    The **SiLU** function is defined as:

    .. math:: \text{SiLU}(x) = x \cdot \frac{1}{1 + \exp(-x)}

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.SiLU(...)`_

    """
    return Function.apply('Silu', input.device, [input])


def smooth_l1_loss(
    input,
    target,
    beta=1.0,
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
    beta : float, optional, default=1.0
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

    See Also
    --------
    `torch.nn.SmoothL1Loss(...)`_

    """
    if size_average is not None or reduce is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return Function.apply(
        'SmoothL1Loss', input.device, [input, target],
        beta=float(beta), reduction=reduction.upper())


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

    See Also
    --------
    `torch.nn.Softmax(...)`_

    """
    return Function.apply(
        'Softmax', input.device, [input],
        outputs=[input if inplace else None], axis=dim)


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

    See Also
    --------
    `torch.nn.SyncBatchNorm(...)`_

    """
    if process_group is None:
        kwargs = locals()
        kwargs.pop('process_group')
        return batch_norm(**kwargs)
    return Function.apply(
        'SyncBatchNorm', input.device,
        [input, weight, bias, running_mean, running_var],
        axis=1, epsilon=eps, use_stats=int(not training),
        momentum=1.0 - momentum, **process_group.arguments)


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

    See Also
    --------
    `torch.nn.Tanh(...)`_

    """
    return Function.apply(
        'Tanh', input.device, [input],
        outputs=[input if inplace else None])


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """Extract the sliding blocks from input.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    kernel_size : Union[int, Sequence[int]]
        The size of sliding window.
    dilation : Union[int, Sequence[int]], optional, default=1
        The dilated rate of sliding window.
    padding : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    stride : Union[int, Sequence[int]], optional, default=1
        The stride of sliding window.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    See Also
    --------
    `torch.nn.Unfold(...)`_

    """
    nd_util = utils._ntuple(input.ndimension() - 2)
    out = Function.apply(
        'Im2Col',
        input.device,
        [input],
        kernel_shape=nd_util(kernel_size),
        strides=nd_util(stride),
        pads=nd_util(padding),
        dilations=nd_util(dilation))
    return out.flatten_(2)


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
    y = F.upsample(x, size=6)  # Shape: (1, 2, 6, 6)
    z = F.upsample(x, scale_factor=2)  # Shape: (1, 2, 6, 8)
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

    See Also
    --------
    `torch.nn.Upsample(...)`_

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

    See Also
    --------
    `torch.nn.UpsamplingBilinear2d(...)`_

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

    See Also
    --------
    `torch.nn.UpsamplingNearest2d(...)`_

    """
    return interpolate(input, size, scale_factor, 'nearest')


def _conv(
    conv_type,
    nd_util,
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=None,
):
    """Apply a conv function."""
    weight_shape = list(weight.shape)
    return Function.apply(
        conv_type,
        input.device,
        [input, weight] + ([bias] if bias else []),
        in_channels=weight_shape[1],
        out_channels=weight_shape[0],
        kernel_shape=weight_shape[2:],
        strides=nd_util(stride),
        pads=nd_util(padding),
        dilations=nd_util(dilation),
        group=groups,
        bias=bias is not None,
        dtype=weight.dtype,
        input_shape=list(input.shape),
    )


def _conv_transpose(
    conv_type,
    nd_util,
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    """Apply a transposed conv function."""
    weight_shape = list(weight.shape)
    return Function.apply(
        conv_type,
        input.device,
        [input, weight] + ([bias] if bias else []),
        in_channels=weight_shape[0],
        out_channels=weight_shape[1],
        kernel_shape=weight_shape[2:],
        strides=nd_util(stride),
        pads=nd_util(padding),
        dilations=nd_util(dilation),
        group=groups,
        output_padding=nd_util(output_padding),
        bias=bias is not None,
        dtype=weight.dtype,
        input_shape=list(input.shape),
    )


def _pool(
    pool_mode,
    nd_util,
    input,
    kernel_size,
    stride=1,
    padding=0,
    ceil_mode=False,
):
    """Apply a pool function."""
    return Function.apply(
        'Pool',
        input.device,
        [input],
        kernel_shape=nd_util(kernel_size),
        strides=nd_util(stride),
        pads=nd_util(padding),
        mode=pool_mode,
        ceil_mode=ceil_mode,
    )
