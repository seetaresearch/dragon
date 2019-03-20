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


def _normalize_tuple(value, rank):
    if not isinstance(value, (list, tuple)): value = [value]
    if len(value) > rank:
        return [value[i] for i in range(rank)]
    else:
        return [value[i] for i in range(len(value))] + \
            [value[-1] for i in range(len(value), rank)]


def _normalize_pads(value, rank):
    if not isinstance(value, (list, tuple)): value = [value]
    if len(value) == (rank * 2): return value
    return _normalize_tuple(value, rank) * 2


@OpSchema.Inputs(2, 3)
def Conv2d(
    inputs, num_output, kernel_shape,
        strides=1, pads=0, dilations=1, group=1,
            padding='VALID', data_format='NCHW', **kwargs):
    """2D Convolution.

    The spatial output dimension of convolution can be computed as follows:

    |conv_output_dim|

    Set ``padding`` to *VALID* will use the value of ``pads``.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [input, weights] + [bias].
    num_output : int
        The output channels of convolution.
    kernel_shape : sequence of int
        The shape of convolution kernel.
    strides : sequence of int, optional, default=1
        The stride(s) of convolution.
    pads : sequence of int, optional, default=0
        The zero padding size(s) of convolution.
    dilations : sequence of int, optional, default=0
        The dilation multiple(s) of convolution.
    group : int, optional, default=1
        The group size of convolution.
    padding : {'VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'}, optional
        The padding algorithm.
    data_format : {'NCHW', 'NHWC'}, optional
        The data_format.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> x = Tensor().Variable()
    >>> weights = Tensor().Normal(std=0.001)
    >>> biases = Tensor().Constant(value=0)
    >>> conv1 = Conv2d([x, weights, biases], num_output=64, kernel_shape=3)

    >>> weights = Tensor().Gaussian(std=0.001)
    >>> conv2 = Conv2d([conv1, weights], num_output=128, kernel_shape=3, strides=1)

    """
    arguments = ParseArgs(locals())

    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: {}'.format(padding))
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    for key in ('kernel_shape', 'strides', 'pads', 'dilations'):
        if key == 'pads': arguments[key] = _normalize_pads(arguments[key], 2)
        else: arguments[key] = _normalize_tuple(arguments[key], 2)

    return Tensor.CreateOperator('Conv2d', **arguments)


@OpSchema.Inputs(2, 3)
def DepthwiseConv2d(
    inputs, num_output, kernel_shape=3, strides=1, pads=0,
        padding='VALID', data_format='NCHW', **kwargs):
    """Depthwise 2D Convolution. `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.

    Set ``padding`` to *VALID* will use the value of ``pads``.

    **Type Constraints**: *float32*

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [input, weights] + [bias].
    num_output : int
        The output channels of convolution.
    kernel_shape : sequence of int, optional, default=3
        The shape of convolution kernel.
    strides : sequence of int, optional, default=1
        The stride(s) of convolution.
    pads : sequence of int, optional, default=0
        The zero padding size(s) of convolution.
    padding : {'VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'}, optional
        The padding algorithm.
    data_format : {'NCHW', 'NHWC'}, optional
        The data_format.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())

    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: {}'.format(padding))
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    for key in ('kernel_shape', 'strides', 'pads', 'dilations'):
        if key == 'pads': arguments[key] = _normalize_pads(arguments[key], 2)
        elif key == 'dilations': arguments[key] = _normalize_tuple([1], 2)
        else: arguments[key] = _normalize_tuple(arguments[key], 2)

    return Tensor.CreateOperator('DepthwiseConv2d', **arguments)


@OpSchema.Inputs(2, 3)
@ArgumentHelper.RepeatedDesc('output_padding')
@ArgumentHelper.RepeatedDesc('output_shape')
def ConvTranspose2d(
    inputs, num_output, kernel_shape,
        strides=1, pads=0, dilations=1, group=1,
            output_padding=None, output_shape=None,
                padding='VALID', data_format='NCHW', **kwargs):
    """2D Deconvolution.

    The spatial output dimension of deconvolution can be computed as follows:

    |deconv_output_dim|

    Set ``padding`` to *VALID* will use the value of ``pads``.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [input, weights] + [bias].
    num_output : int
        The output channels of deconvolution.
    kernel_shape : sequence of int
        The shape of convolution kernel.
    strides : sequence of int, optional, default=1
        The stride(s) of deconvolution.
    pads : sequence of int, optional, default=0
        The zero padding size(s) of deconvolution.
    dilations : sequence of int, optional, default=1
        The dilation multiple(s) of deconvolution.
    group : int, optional, default=1
        The group size of deconvolution.
    output_padding : sequence of (int, Tensor), optional
        The padding value add to one side(right) of the output.
    output_shape : sequence of (int, Tensor), optional
        The deterministic output shape for **SAME** padding.
    padding : {'VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'}, optional
        The padding algorithm.
    data_format : {'NCHW', 'NHWC'}, optional
        The data_format.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> input = Tensor().Variable()
    >>> weights = Tensor().Normal(std=0.001)
    >>> biases = Tensor().Constant(value=0)
    >>> deconv1 = ConvTranspose2d([input, weights, biases], num_output=64, kernel_shape=3)

    >>> weights = Tensor().Gaussian(std=0.001)
    >>> deconv2 = ConvTranspose2d([deconv1, weights], num_output=128, kernel_shape=3, strides=1)

    """
    arguments = ParseArgs(locals())

    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: {}'.format(padding))
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if output_padding is not None or output_shape is not None:
        if 'SAME' not in arguments['padding']:
            arguments['padding'] = 'SAME_LOWER' # Enforce the auto padding

    for key in ('kernel_shape', 'strides', 'pads', 'dilations'):
        if key == 'pads': arguments[key] = _normalize_pads(arguments[key], 2)
        else: arguments[key] = _normalize_tuple(arguments[key], 2)

    return Tensor.CreateOperator('ConvTranspose2d', **arguments)


@OpSchema.Inputs(1)
def Pool2d(
    inputs, kernel_shape, strides, pads=0, padding='VALID', ceil_mode=True,
        mode='MAX', data_format='NCHW', global_pooling=False, **kwargs):
    """2D Pooling, MAX or AVG.

    The spatial output dimension of pooling can be computed as follows:

    |pooling_output_dim|

    Set ``padding`` to *VALID* will use the value of ``pads``.

    If ``global_pooling`` is *True*, ``strides`` and ``pads`` will be set to *1* and *0* respectively.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    kernel_shape : int, tuple or list
        The shape of pooling kernel.
    strides : sequence of int
        The stride(s) of of pooling,
    pads : sequence of int, optional, default=0
        The zero padding size(s) of pooling.
    padding : {'VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'}, optional
        The padding algorithm.
    ceil_mode : bool, optional, default=True
        Whether to ceil the boundary.
    mode : {'MAX', 'AVG'}, optional
        The pooling mode.
    data_format : {'NCHW', 'NHWC'}, optional
        The data_format.
    global_pooling : bool, optional
        Whether to use global pooling.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())

    if mode not in ('MAX', 'AVG'):
        raise ValueError('Unsupported lrn mode: {}'.format(mode))
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: {}'.format(padding))
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    for key in ('kernel_shape', 'strides', 'pads'):
        if key == 'pads': arguments[key] = _normalize_pads(arguments[key], 2)
        else: arguments[key] = _normalize_tuple(arguments[key], 2)

    return Tensor.CreateOperator('Pool2d', **arguments)


@OpSchema.Inputs(2)
def ROIPool(inputs, pool_h, pool_w, spatial_scale=1.0, **kwargs):
    """Max RoIPooling. `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent the Feature and RoIs respectively.
    pool_h : int, optional
        The height of pooled tensor.
    pool_w : int, optional
        The width of pooled tensor.
    spatial_scale : float, optional
        The ``inverse`` of total down-sampling multiples on input tensor.

    Returns
    -------
    Tensor
        The batch of pooled RoI regions.

    """
    return Tensor.CreateOperator('ROIPool', **ParseArgs(locals()))


@OpSchema.Inputs(2)
def ROIAlign(inputs, pool_h=0, pool_w=0, spatial_scale=1.0, sampling_ratio=2, **kwargs):
    """AVG RoIAlign. `[He et.al, 2017] <https://arxiv.org/abs/1703.06870>`_.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent the Feature and RoIs respectively.
    pool_h : int, optional
        The height of pooled tensor.
    pool_w : int, optional
        The width of pooled tensor.
    spatial_scale : float, optional
        The ``inverse`` of total down-sampling multiples on input tensor.
    sampling_ratio : int, optional
        The number of sampling grids for each RoI bin.

    Returns
    -------
    Tensor
        The batch of pooled RoI regions.

    """
    return Tensor.CreateOperator('ROIAlign', **ParseArgs(locals()))


@OpSchema.Inputs(1)
def LRN(
    inputs, local_size=5, alpha=0.0001, beta=0.75, k=2.0,
        mode='ACROSS_CHANNELS', data_format='NCHW', **kwargs):
    """Local Response Normalization. `[Krizhevsky et.al, 2012] <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>`_.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    local_size : int, optional
        The local size of LRN.
    alpha : float, optional
        The alpha of LRN.
    beta : float, optional
        The beta of LRN.
    k : float, optional
        The k of LRN.
    mode : {'ACROSS_CHANNELS', 'WITHIN_CHANNEL'}, optional
        The lrn mode.
    data_format : {'NCHW', 'NHWC'}, optional
        The data_format.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())

    if mode not in ('ACROSS_CHANNELS', 'WITHIN_CHANNEL'):
        raise ValueError('Unsupported lrn mode: {}'.format(mode))
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return Tensor.CreateOperator('LRN', **arguments)


@OpSchema.Inputs(1)
@ArgumentHelper.RepeatedDesc('dsize')
def NNResize(
    inputs, dsize, shape_like=None,
        fy=-1.0, fx=-1.0, data_format='NCHW', **kwargs):
    """Resize the image with Nearest-Neighbor method.

    Set ``dsize`` to None if you want to use ``shape_like`` or ``fy/fx``.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    dsize : sequence of (int, Tensor)
        The output size, formats as (h, w).
    shape_like : Tensor, optional
        The tensor for guiding the shape of resizing.
    fy : float, optional, default=-1.0
        The scale factor based on src height.
    fx : float, optional, default=-1.0
        The scale factor based on src width.
    data_format : {'NCHW', 'NHWC'}, optional
        The data_format.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())

    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if dsize is not None and len(dsize) != 2:
        raise ValueError('The dsize should be a list with 2 elements.')

    if shape_like is not None:
        if not isinstance(shape_like, Tensor):
            raise TypeError('The shape_like should be a Tensor.')
        arguments['shape_like'] = shape_like.name

    if dsize is None and shape_like is None and (fy == -1.0 or fx == -1.0):
        raise RuntimeError('The dsize, shape_like or fy/fx should be specified either.')

    return Tensor.CreateOperator('NNResize', **arguments)


@OpSchema.Inputs(1)
@ArgumentHelper.RepeatedDesc('dsize')
def BilinearResize(
    inputs, dsize, shape_like=None,
        fy=-1.0, fx=-1.0, data_format='NCHW', **kwargs):
    """Resize the image with Bi-linear method.

    Set ``dsize`` to None if you want to use ``shape_like`` or ``fy/fx``.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    dsize : sequence of (int, Tensor)
        The output size, formats as (h, w).
    shape_like : Tensor, optional
        The tensor for guiding the shape of resizing.
    fy : float, optional, default=-1.0
        The scale factor based on src height.
    fx : float, optional, default=-1.0
        The scale factor based on src width.
    data_format : {'NCHW', 'NHWC'}, optional
        The data_format.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())

    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if dsize is not None and len(dsize) != 2:
        raise ValueError('The dsize should be a list with 2 elements.')

    if shape_like is not None:
        if not isinstance(shape_like, Tensor):
            raise TypeError('The shape_like should be a Tensor.')
        arguments['shape_like'] = shape_like.name

    if dsize is None and shape_like is None and (fy == -1.0 or fx == -1.0):
        raise RuntimeError('The dsize, shape_like or fy/fx should be specified either.')

    return Tensor.CreateOperator('BilinearResize', **arguments)


@OpSchema.Inputs(2)
def BiasAdd(inputs, data_format='NCHW', **kwargs):
    """Add the bias across channels to a ``NCHW`` or ``NHWC`` input.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent [input, bias].
    data_format : {'NCHW', 'NHWC'}, optional
        The data_format.

    Returns
    -------
    Tensor
        A bias added tensor.

    """
    arguments = ParseArgs(locals())

    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return Tensor.CreateOperator('BiasAdd', **arguments)


@OpSchema.Inputs(1)
@ArgumentHelper.Desc('keep_prob', as_target=False)
def DropBlock2d(
    inputs, block_size=7, keep_prob=0.9, alpha=1.,
        decrement=0., data_format='NCHW', **kwargs):
    """Randomly drop the outputs according to the spatial blocks. `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

    Set the ``decrement`` to schedule ``keep_prob`` for each iteration.

    Set the ``alpha`` to decrease ``gamma`` for different stages.

    **Type Constraints**: (*float16*, *float32*)

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    block_size : int, optional
        The size of dropping block.
    keep_prob : float or Tensor, optional, default=0.9
        The prob of keeping.
    alpha : float, optional, default=1.0
        The scale factor to gamma.
    decrement : float, optional, default=0.0
        The decrement to keep prob.
    data_format : {'NCHW', 'NHWC'}, optional
        The data_format.

    Returns
    -------
    Tensor
        The output tensor.

    """
    return Tensor.CreateOperator('DropBlock2d', **ParseArgs(locals()))