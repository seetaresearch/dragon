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
"""Vision ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.eager import context
from dragon.core.ops import vision_ops_lib
from dragon.core.ops.utils import ArgHelper
from dragon.core.ops.utils import OpSchema
from dragon.core.util import nest


@OpSchema.num_inputs(2)
def bias_add(inputs, data_format='NCHW', **kwargs):
    """Add the bias across channels to input.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The ``input`` and ``bias``.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    op_lib = vision_ops_lib.BiasAdd
    if context.executing_eagerly():
        return op_lib \
            .instantiate(data_format=data_format) \
            .apply(inputs, args.get('inplace', False))
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2, 3)
def conv2d(
    inputs,
    kernel_shape=3,
    strides=1,
    pads=0,
    dilations=1,
    group=1,
    padding='VALID',
    data_format='NCHW',
    **kwargs
):
    r"""Apply the 2d convolution.

    Set ``padding`` to **VALID** will use the value of ``pads``.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and ``bias``.
    kernel_shape : Sequence[int], optional, default=3
        The shape of convolution kernel.
    strides : Sequence[int], optional, default=1
        The stride(s) of sliding window.
    pads : Sequence[int], optional, default=0
        The zero-padding size(s).
    dilations : Sequence[int], optional, default=1
        The rate(s) of dilated kernel.
    group : int, optional, default=1
        The number of groups to split input channels.
    padding : {'VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'}, optional
        The padding algorithm.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: %s' % padding)
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    for key in ('kernel_shape', 'strides', 'pads', 'dilations'):
        if key == 'pads':
            args[key] = _normalize_pads(args[key], 2)
        else:
            args[key] = _normalize_tuple(args[key], 2)
    op_lib = vision_ops_lib.Conv2d
    if context.executing_eagerly():
        weight_shape = inputs[1].shape
        return op_lib \
            .instantiate(
                dim_in=weight_shape[1],
                dim_out=weight_shape[0],
                kernel_shape=args['kernel_shape'],
                strides=args['strides'],
                pads=args['pads'],
                dilations=args['dilations'],
                group=group,
                padding=padding,
                data_format=data_format,
                bias=len(inputs) > 2,
                dtype=inputs[1].dtype,
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2, 3)
@ArgHelper.repeated_desc('output_padding')
@ArgHelper.repeated_desc('output_shape')
def conv2d_transpose(
    inputs,
    kernel_shape=3,
    strides=1,
    pads=0,
    dilations=1,
    group=1,
    output_padding=None,
    output_shape=None,
    padding='VALID',
    data_format='NCHW',
    **kwargs
):
    r"""Apply the 2d deconvolution.

    Set ``padding`` to **VALID** will use the value of ``pads``.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and ``bias``.
    kernel_shape : Sequence[int], optional, default=3
        The shape of convolution kernel.
    strides : Sequence[int], optional, default=1
        The stride(s) of sliding window.
    pads : Sequence[int], optional, default=0
        The zero padding size(s).
    dilations : Sequence[int], optional, default=1
        The rate(s) of dilated kernel.
    group : int, optional, default=1
        The group size of convolution.
    output_padding : Sequence[Union[int, dragon.Tensor]], optional
        The extra size padding to output.
    output_shape : Sequence[Union[int, dragon.Tensor]], optional
        The output shape for **SAME** padding.
    padding : {'VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'}, optional
        The padding algorithm.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: %s' % padding)
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    if 'SAME' in padding and output_shape is None:
        raise ValueError('Excepted <output_shape> for same padding.')
    if output_shape is not None and 'SAME' not in padding:
        args['padding'] = 'SAME'
    for key in ('kernel_shape', 'strides', 'pads', 'dilations'):
        if key in args and args[key] is not None:
            if key == 'pads':
                args[key] = _normalize_pads(args[key], 2)
            else:
                args[key] = _normalize_tuple(args[key], 2)
    op_lib = vision_ops_lib.ConvTranspose2d
    if context.executing_eagerly():
        weight_shape = inputs[1].shape
        return op_lib \
            .instantiate(
                dim_in=weight_shape[0],
                dim_out=weight_shape[1],
                kernel_shape=args['kernel_shape'],
                strides=args['strides'],
                pads=args['pads'],
                dilations=args['dilations'],
                group=group,
                padding=args['padding'],
                output_padding=args['output_padding'],
                output_shape=args['output_shape'],
                data_format=data_format,
                bias=len(inputs) > 2,
                dtype=inputs[1].dtype,
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2, 3)
def depthwise_conv2d(
    inputs,
    kernel_shape=3,
    strides=1,
    pads=0,
    dilations=1,
    padding='VALID',
    data_format='NCHW',
    **kwargs
):
    r"""Apply the 2d depthwise convolution.
    `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.

    Set ``padding`` to **VALID** will use the value of ``pads``.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and ``bias``.
    kernel_shape : Sequence[int], optional, default=3
        The size(s) of convolution kernel.
    strides : Sequence[int], optional, default=1
        The stride(s) of sliding window.
    pads : Sequence[int], optional, default=0
        The zero padding size(s) of convolution.
    dilations : Sequence[int], optional, default=0
        The rate(s) of dilated kernel.
    padding : {'VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'}, optional
        The padding algorithm.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: %s' % padding)
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    for key in ('kernel_shape', 'strides', 'pads', 'dilations'):
        if key == 'pads':
            args[key] = _normalize_pads(args[key], 2)
        else:
            args[key] = _normalize_tuple(args[key], 2)
    op_lib = vision_ops_lib.DepthwiseConv2d
    if context.executing_eagerly():
        weight_shape = inputs[1].shape
        return op_lib \
            .instantiate(
                dim_in=weight_shape[1],
                dim_out=weight_shape[0],
                kernel_shape=args['kernel_shape'],
                strides=args['strides'],
                pads=args['pads'],
                dilations=args['dilations'],
                padding=padding,
                data_format=data_format,
                bias=len(inputs) > 2,
                dtype=inputs[1].dtype,
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def depth_to_space(inputs, block_size, data_format='NCHW', **kwargs):
    """Rearrange depth data into spatial blocks.

    Examples:

    ```python
    n, c, h, w, bs = 1, 4, 1, 1, 2
    x = dragon.arange(n * c * h * w).reshape((n, c, h, w))
    y = dragon.reshape(x, (n, bs, bs, c // (bs ** 2), h, w))
    y = dragon.transpose(y, (0, 3, 4, 1, 5, 2))
    y = dragon.reshape(y, (n, c // (bs ** 2), h * bs, w * bs))
    z = dragon.nn.depth_to_space(x, 2)  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    block_size : int, required
        The size of spatial block.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    op_lib = vision_ops_lib.DepthToSpace
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                block_size=block_size,
                data_format=data_format,
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def pool2d(
    inputs,
    kernel_shape,
    strides,
    pads=0,
    padding='VALID',
    ceil_mode=False,
    mode='MAX',
    data_format='NCHW',
    global_pooling=False,
    **kwargs
):
    r"""Apply the 2d pooling.

    Set ``padding`` to **VALID** will use the value of ``pads``.

    If ``global_pooling`` is **True**, ``strides`` and ``pads`` will be set to **1** and **0**.

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    kernel_shape : Sequence[int]
        The shape of pooling kernel.
    strides : Sequence[int]
        The stride(s) of of pooling,
    pads : Sequence[int], optional, default=0
        The zero padding size(s) of pooling.
    padding : {'VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'}, optional
        The padding algorithm.
    ceil_mode : bool, optional, default=False
        Whether to ceil the boundary.
    mode : {'MAX', 'AVG'}, optional
        The pooling mode.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.
    global_pooling : bool, optional, default=False
        Whether to apply the global pooling.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['mode'] = mode.upper()
    if args['mode'] not in ('MAX', 'AVG'):
        raise ValueError('Unsupported pooling mode: %s' % mode)
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: %s' % padding)
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    for key in ('kernel_shape', 'strides', 'pads'):
        if key == 'pads':
            args[key] = _normalize_pads(args[key], 2)
        else:
            args[key] = _normalize_tuple(args[key], 2)
    op_lib = vision_ops_lib.Pool2d
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                kernel_shape=args['kernel_shape'],
                strides=args['strides'],
                pads=args['pads'],
                padding=padding,
                ceil_mode=ceil_mode,
                mode=args['mode'],
                data_format=data_format,
                global_pooling=global_pooling,
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
@ArgHelper.repeated_desc('sizes')
@ArgHelper.repeated_desc('scales')
def resize(
    inputs,
    sizes=None,
    scales=None,
    mode='linear',
    align_corners=False,
    data_format='NCHW',
    **kwargs
):
    r"""Resize input via interpolating neighborhoods.

    ``sizes`` or ``scales`` will be selected by ``data_format``:

    ```python
    x, sizes = dragon.ones((1, 2, 3, 4)), (6, 6)
    a = dragon.vision.resize(x, sizes, data_format='NCHW')  # Shape: (1, 2, 6, 6)
    c = dragon.vision.resize(x, sizes, data_format='NHWC')  # Shape: (1, 6, 6, 4)
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
    inputs : dragon.Tensor
        The input tensor.
    sizes : Union[int, Sequence[int], dragon.Tensor], optional
        The output dimensions.
    scales : Union[float, Sequence[float], dragon.Tensor], optional
        The scale along each input dimension.
    mode : {'nearest', 'linear'}, optional
        The interpolation mode.
    align_corners : bool, optional, default=False
        Whether to align corners in linear interpolating.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['mode'] = mode.upper()
    if sizes is None and scales is None:
        raise ValueError('Specify either <sizes> or <scales>.')
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))
    op_lib = vision_ops_lib.Resize
    if context.executing_eagerly():
        if sizes is not None:
            args['sizes'] = nest.flatten(args['sizes'])
        if scales is not None:
            args['scales'] = nest.flatten(args['scales'])
        return op_lib \
            .instantiate(
                mode=args['mode'],
                align_corners=align_corners,
                num_sizes=len(args['sizes']) if sizes is not None else 0,
                num_scales=len(args['scales']) if scales is not None else 0,
                data_format=data_format,
            ).apply([inputs], args['sizes'], args['scales'])
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2)
def roi_align(
    inputs,
    pooled_h,
    pooled_w,
    spatial_scale=1.,
    sampling_ratio=2,
    **kwargs
):
    r"""Apply the average roi align.
    `[He et.al, 2017] <https://arxiv.org/abs/1703.06870>`_.

    The **rois** should be packed with a shape like :math:`(N, 5)`,
    where :math:`N` is the number of RoIs.

    Each RoI is a 5d sequence containing **(batch_index, x1, y1, x2, y2)**.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``feature`` and ``rois``.
    pooled_h : int, required
        The output height.
    pooled_w : int, required
        The output width.
    spatial_scale : float, optional, default=1.
        The input scale to the size of ``rois``.
    sampling_ratio : int, optional, default=2
        The number of sampling grids for ``rois``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['spatial_scale'] = float(spatial_scale)
    op_lib = vision_ops_lib.RoiAlign
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                pooled_h=pooled_h,
                pooled_w=pooled_w,
                spatial_scale=args['spatial_scale'],
                sampling_ratio=sampling_ratio,
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(2)
def roi_pool(
    inputs,
    pooled_h,
    pooled_w,
    spatial_scale=1.,
    **kwargs
):
    r"""Apply the max roi pooling.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    The **rois** should be packed with a shape like :math:`(N, 5)`,
    where :math:`N` is the number of RoIs.

    Each RoI is a 5d sequence containing **(batch_index, x1, y1, x2, y2)**.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``feature`` and ``rois``.
    pooled_h : int, required
        The output height.
    pooled_w : int, required
        The output width.
    spatial_scale : float, optional, default=1.
        The input scale to the size of ``rois``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    args['spatial_scale'] = float(spatial_scale)
    op_lib = vision_ops_lib.RoiPool
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                pooled_h=pooled_h,
                pooled_w=pooled_w,
                spatial_scale=args['spatial_scale'],
            ).apply(inputs)
    else:
        return op_lib.blend(**args)


@OpSchema.num_inputs(1)
def space_to_depth(inputs, block_size, data_format='NCHW', **kwargs):
    """Rearrange blocks of spatial data into depth.

    Examples:

    ```python
    n, c, h, w, bs = 1, 2, 2, 2, 2
    x = dragon.arange(n * c * h * w).reshape((n, c, h, w))
    y = dragon.reshape(x, (n, c, h // bs, bs, w // bs, bs))
    y = dragon.transpose(y, (0, 3, 5, 1, 2, 4))
    y = dragon.reshape(y, (n, c * (bs ** 2), h // bs, w // bs))
    z = dragon.nn.space_to_depth(x, 2)  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    block_size : int, required
        The size of spatial block.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = ArgHelper.parse(locals())
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    op_lib = vision_ops_lib.SpaceToDepth
    if context.executing_eagerly():
        return op_lib \
            .instantiate(
                block_size=block_size,
                data_format=data_format,
            ).apply([inputs])
    else:
        return op_lib.blend(**args)


def _normalize_tuple(value, rank):
    """Repeat the value to a tuple."""
    value = nest.flatten(value)
    if len(value) > rank:
        return [value[i] for i in range(rank)]
    else:
        return [value[i] for i in range(len(value))] + \
               [value[-1] for _ in range(len(value), rank)]


def _normalize_pads(value, rank):
    """Repeat the value to a padding tuple."""
    value = nest.flatten(value)
    if len(value) == (rank * 2):
        return value
    return _normalize_tuple(value, rank) * 2
