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
"""NN ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.core.framework import types
from dragon.core.ops import activation_ops
from dragon.core.ops import loss_ops
from dragon.core.ops import math_ops
from dragon.core.ops import normalization_ops
from dragon.core.ops import sort_ops
from dragon.core.ops import vision_ops
from dragon.core.util import nest
from dragon.core.util import six


def avg_pool(
    input,
    ksize,
    strides,
    padding='VALID',
    data_format='NHWC',
    name=None,
):
    r"""Apply the n-dimension average pooling.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, D1, D2, ...)`, and output shape is
      :math:`(N, C, D1_{\text{out}}, D2_{\text{out}}, ...)`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D1, D2, ..., C)`, and output shape is
      :math:`(N, D1_{\text{out}}, D2_{\text{out}}, ..., C)`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    for i in range(3):
        ndim = i + 1
        x = tf.ones((1,) + (2,) * ndim + (2,))
        y = tf.nn.avg_pool(x, ksize=(2,) * ndim, strides=2)
        assert y.shape == (1,) + (1,) * ndim + (2,)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Union[int, Sequence[int]]
        The size of pooling window.
    strides : Union[int, Sequence[int]]
        The stride of pooling window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if input.shape is not None:
        num_total_dims = len(input.shape)
    else:
        num_total_dims = len(ksize)
    num_spatial_dims = num_total_dims - 2
    data_format = data_format if data_format else 'NHWC'
    start_axis = 2 if data_format.startswith('NC') else 1
    normalize_spatial_args = functools.partial(
        _normalize_spatial_args,
        num_total_dims=num_total_dims,
        num_spatial_dims=num_spatial_dims,
        start_axis=start_axis)
    ksize = normalize_spatial_args('ksize', ksize)
    strides = normalize_spatial_args('strides', strides)
    padding, pads = normalize_spatial_args('padding', padding)
    return getattr(vision_ops, 'pool{}d'.format(num_spatial_dims))(
        input,
        kernel_shape=ksize[start_axis:start_axis + num_spatial_dims],
        strides=strides[start_axis:start_axis + num_spatial_dims],
        padding=padding,
        pads=pads,
        mode='avg',
        data_format='NCHW' if data_format.startswith('NC') else 'NHWC',
        name=name,
    )


def avg_pool1d(
    input,
    ksize,
    strides,
    padding='VALID',
    data_format='NHWC',
    name=None,
):
    r"""Apply the 1d average pooling.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, H)`, and output shape is :math:`(N, C, H_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, C)`, and output shape is :math:`(N, H_{\text{out}}, C)`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2))
    y = tf.nn.avg_pool1d(x, ksize=2, strides=2)
    assert y.shape == (1, 1, 2)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Union[int, Sequence[int]]
        The size of pooling window.
    strides : Union[int, Sequence[int]]
        The stride of pooling window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    start_axis = 2 if data_format.startswith('NC') else 1
    return avg_pool(
        input=input,
        ksize=_normalize_spatial_args('ksize', ksize, 3, 1, start_axis),
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )


def avg_pool2d(
    input,
    ksize,
    strides,
    padding='VALID',
    data_format='NHWC',
    name=None,
):
    r"""Apply the 2d average pooling.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, H, W)`, and output shape is
      :math:`(N, C, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, C)`, and output shape is
      :math:`(N, H_{\text{out}}, W_{\text{out}}, C)`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2, 2))
    y = tf.nn.avg_pool2d(x, ksize=2, strides=2)
    assert y.shape == (1, 1, 1, 2)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Union[int, Sequence[int]]
        The size of pooling window.
    strides : Union[int, Sequence[int]]
        The stride of pooling window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    start_axis = 2 if data_format.startswith('NC') else 1
    return avg_pool(
        input=input,
        ksize=_normalize_spatial_args('ksize', ksize, 4, 2, start_axis),
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )


def avg_pool3d(
    input,
    ksize,
    strides,
    padding='VALID',
    data_format='NHWC',
    name=None,
):
    r"""Apply the 3d average pooling.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, D, H, W)`, and output shape is
      :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D, H, W, C)`, and output shape is
      :math:`(N, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}, C)`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2, 2, 2))
    y = tf.nn.avg_pool3d(x, ksize=2, strides=2)
    assert y.shape == (1, 1, 1, 1, 2)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Union[int, Sequence[int]]
        The size of pooling window.
    strides : Union[int, Sequence[int]]
        The stride of pooling window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    start_axis = 2 if data_format.startswith('NC') else 1
    return avg_pool(
        input=input,
        ksize=_normalize_spatial_args('ksize', ksize, 5, 3, start_axis),
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )


def bias_add(value, bias, data_format='NHWC', name=None):
    """Add the bias across channels to input.

    Parameters
    ----------
    value : dragon.Tensor
        The input tensor.
    bias : dragon.Tensor
        The bias tensor.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return vision_ops.bias_add([value, bias], data_format=data_format, name=name)


def convolution(
    input,
    filters,
    strides=1,
    padding='VALID',
    data_format='NHWC',
    dilations=1,
    name=None,
    **kwargs
):
    r"""Apply the n-dimension convolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, D1, D2, ...)`, filters shape
      :math:`(C_{\text{out}}, C_{\text{in}}, D1_{\text{f}}, D2_{\text{f}}, ...)`,
      and output shape is :math:`(N, C_{\text{out}}, D1_{\text{out}}, D2_{\text{out}}, ...)`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D1, D2, ..., C_{\text{in}})`, filters shape
      :math:`(C_{\text{out}}, D1_{\text{f}}, D2_{\text{f}}, ..., C_{\text{in}})`,
      and output shape is :math:`(N, D1_{\text{out}}, D2_{\text{out}}, ..., C_{\text{out}})`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    for i in range(3):
        ndim = i + 1
        x = tf.ones((1,) + (2,) * ndim + (2,))
        filters = tf.ones((3,) + (1,) * ndim + (2,))
        y = tf.nn.convolution(x, filters)
        assert y.shape == (1,) + (2,) * ndim + (3,)
    ```

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The filters tensor.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated filters.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    num_total_dims = len(filters.shape)
    num_spatial_dims = num_total_dims - 2
    data_format = data_format if data_format else 'NHWC'
    channel_axis = 1 if data_format.startswith('NC') else -1
    start_axis = 2 if data_format.startswith('NC') else 1
    normalize_spatial_args = functools.partial(
        _normalize_spatial_args,
        num_total_dims=num_total_dims,
        num_spatial_dims=num_spatial_dims,
        start_axis=start_axis)
    strides = normalize_spatial_args('strides', strides)
    dilations = normalize_spatial_args('dilations', dilations)
    padding, pads = normalize_spatial_args('padding', padding)
    return getattr(vision_ops, '{}{}d'.format(
        kwargs.get('conv_type', 'conv'), num_spatial_dims))(
            [input, filters],
            kernel_shape=filters.shape[-num_spatial_dims:],
            strides=strides[start_axis:start_axis + num_spatial_dims],
            dilations=dilations[start_axis:start_axis + num_spatial_dims],
            padding=padding,
            pads=pads,
            group=input.shape[channel_axis] // filters.shape[1],
            data_format='NCHW' if data_format.startswith('NC') else 'NHWC',
            name=name)


def conv_transpose(
    input,
    filters,
    output_shape=None,
    strides=1,
    padding='SAME',
    output_padding=None,
    data_format='NHWC',
    dilations=1,
    name=None,
):
    r"""Apply the n-dimension deconvolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, D1, D2, ...)`, filters shape
      :math:`(C_{\text{in}}, C_{\text{out}}, D1_{\text{f}}, D2_{\text{f}}, ...)`,
      and output shape is :math:`(N, C_{\text{out}}, D1_{\text{out}}, D2_{\text{out}}, ...)`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D1, D2, ..., C_{\text{in}})`, filters shape
      :math:`(C_{\text{in}}, D1_{\text{f}}, D2_{\text{f}}, ..., C_{\text{out}})`,
      and output shape is :math:`(N, D1_{\text{out}}, D2_{\text{out}}, ..., C_{\text{out}})`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    for i in range(3):
        ndim = i + 1
        x = tf.ones((1,) + (2,) * ndim + (2,))
        filters = tf.ones((3,) + (1,) * ndim + (2,))
        y = tf.nn.conv_transpose(x, filters, output_shape=(2,) * ndim)
        assert y.shape == (1,) + (2,) * ndim + (3,)
    ```

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The filters tensor.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The optional output shape.
    strides : Union[int, Sequence[int]], default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int], str], optional
        The padding algorithm or size.
    output_padding : Union[Sequence[int], dragon.Tensor], optional
        The additional size added to the output shape.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated filters.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if filters.shape is not None:
        num_total_dims = len(filters.shape)
    elif input.shape is not None:
        num_total_dims = len(input.shape)
    else:
        raise ValueError('Rank of <input> or <filters> must be known.')
    num_spatial_dims = num_total_dims - 2
    data_format = data_format if data_format else 'NHWC'
    start_axis = 2 if data_format.startswith('NC') else 1
    normalize_spatial_args = functools.partial(
        _normalize_spatial_args,
        num_total_dims=num_total_dims,
        num_spatial_dims=num_spatial_dims,
        start_axis=start_axis)
    strides = normalize_spatial_args('strides', strides)
    dilations = normalize_spatial_args('dilations', dilations)
    padding, pads = normalize_spatial_args('padding', padding)
    if padding == 'SAME' and output_shape is None:
        raise ValueError('Excepted <output_shape> for same padding.')
    output_shape = normalize_spatial_args('output_shape', output_shape)
    return getattr(vision_ops, 'conv{}d_transpose'.format(num_spatial_dims))(
        [input, filters],
        kernel_shape=filters.shape[start_axis:start_axis + num_spatial_dims],
        strides=strides[start_axis:start_axis + num_spatial_dims],
        dilations=dilations[start_axis:start_axis + num_spatial_dims],
        padding=padding,
        output_padding=output_padding,
        output_shape=output_shape,
        pads=pads,
        data_format='NCHW' if data_format.startswith('NC') else 'NHWC',
        name=name,
    )


def conv1d(
    input,
    filters,
    strides=1,
    padding='VALID',
    data_format='NHWC',
    dilations=None,
    name=None,
):
    r"""Apply the 1d convolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H)`, filters shape
      :math:`(C_{\text{out}}, C_{\text{in}}, H_{\text{f}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, C_{\text{in}})`, filters shape
      :math:`(C_{\text{out}}, H_{\text{f}}, C_{\text{in}})`,
      and output shape is :math:`(N, H_{\text{out}}, C_{\text{out}})`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2))
    filters = tf.ones((3, 1, 2))
    y = tf.nn.conv1d(x, filters)
    assert y.shape == (1, 2, 3)
    ```

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The filters tensor.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated filters.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return convolution(**locals())


def conv2d(
    input,
    filters,
    strides=1,
    padding='VALID',
    data_format='NHWC',
    dilations=None,
    name=None,
):
    r"""Apply the 2d convolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H, W)`, filters shape
      :math:`(C_{\text{out}}, C_{\text{in}}, H_{\text{f}}, W_{\text{f}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, C_{\text{in}})`, filters shape
      :math:`(C_{\text{out}}, H_{\text{f}}, W_{\text{f}}, C_{\text{in}})`,
      and output shape is :math:`(N, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2, 2))
    filters = tf.ones((3, 1, 1, 2))
    y = tf.nn.conv2d(x, filters)
    assert y.shape == (1, 2, 2, 3)
    ```

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The filters tensor.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated filters.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return convolution(**locals())


def conv3d(
    input,
    filters,
    strides,
    padding,
    data_format='NHWC',
    dilations=1,
    name=None,
):
    r"""Apply the 3d convolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, D, H, W)`, filters shape
      :math:`(C_{\text{out}}, C_{\text{in}}, D_{\text{f}}, H_{\text{f}}, W_{\text{f}})`,
      and output shape is :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D, H, W, C_{\text{in}})`, filters shape
      :math:`(C_{\text{out}}, D_{\text{f}}, H_{\text{f}}, W_{\text{f}}, C_{\text{in}})`,
      and output shape is :math:`(N, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2, 2, 2))
    filters = tf.ones((3, 1, 1, 1, 2))
    y = tf.nn.conv3d(x, filters)
    assert y.shape == (1, 2, 2, 2, 3)
    ```

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The filters tensor.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated filters.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return convolution(**locals())


def conv1d_transpose(
    input,
    filters,
    output_shape=None,
    strides=1,
    padding='SAME',
    output_padding=None,
    data_format='NHWC',
    dilations=None,
    name=None,
):
    r"""Apply the 1d deconvolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H)`, filters shape
      :math:`(C_{\text{in}}, C_{\text{out}}, H_{\text{f}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, C_{\text{in}})`, filters shape
      :math:`(C_{\text{in}}, H_{\text{f}}, C_{\text{out}})`,
      and output shape is :math:`(N, H_{\text{out}}, C_{\text{out}})`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2))
    filters = tf.ones((3, 1, 2))
    y = tf.nn.conv1d_transpose(x, filters, output_shape=(2,))
    assert y.shape == (1, 2, 3)
    ```

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The filters tensor.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The optional output shape.
    strides : Union[int, Sequence[int]], default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int], str]
        The padding algorithm or size.
    output_padding : Union[Sequence[int], dragon.Tensor], optional
        The additional size added to the output shape.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated filters.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return conv_transpose(**locals())


def conv2d_transpose(
    input,
    filters,
    output_shape=None,
    strides=1,
    padding='SAME',
    output_padding=None,
    data_format='NHWC',
    dilations=None,
    name=None,
):
    r"""Apply the 2d deconvolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H, W)`, filters shape
      :math:`(C_{\text{in}}, C_{\text{out}}, H_{\text{f}}, W_{\text{f}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, C_{\text{in}})`, filters shape
      :math:`(C_{\text{in}}, H_{\text{f}}, W_{\text{f}}, C_{\text{out}})`,
      and output shape is :math:`(N, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2, 2))
    filters = tf.ones((3, 1, 1, 2))
    y = tf.nn.conv2d_transpose(x, filters, output_shape=(2, 2))
    assert y.shape == (1, 2, 2, 3)
    ```

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The filters tensor.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The optional output shape.
    strides : Union[int, Sequence[int]], default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int], str]
        The padding algorithm or size.
    output_padding : Union[Sequence[int], dragon.Tensor], optional
        The additional size added to the output shape.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated filters.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return conv_transpose(**locals())


def conv3d_transpose(
    input,
    filters,
    output_shape=None,
    strides=1,
    padding='SAME',
    output_padding=None,
    data_format='NHWC',
    dilations=None,
    name=None,
):
    r"""Apply the 3d deconvolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, D, H, W)`, filters shape
      :math:`(C_{\text{in}}, C_{\text{out}}, D_{\text{f}}, H_{\text{f}}, W_{\text{f}})`,
      and output shape is :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D, H, W, C_{\text{in}})`, filters shape
      :math:`(C_{\text{in}}, D_{\text{f}}, H_{\text{f}}, W_{\text{f}}, C_{\text{out}})`,
      and output shape is :math:`(N, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2, 2, 2))
    filters = tf.ones((3, 1, 1, 1, 2))
    y = tf.nn.conv3d_transpose(x, filters, output_shape=(2, 2, 2))
    assert y.shape == (1, 2, 2, 2, 3)
    ```

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The filters tensor.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The optional output shape.
    strides : Union[int, Sequence[int]], default=1
        The stride of convolution window.
    padding : Union[int, Sequence[int], str]
        The padding algorithm or size.
    output_padding : Union[Sequence[int], dragon.Tensor], optional
        The additional size added to the output shape.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated filters.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return conv_transpose(**locals())


def depthwise_conv2d(
    input,
    filters,
    strides=1,
    padding='VALID',
    data_format='NHWC',
    dilations=1,
    name=None,
):
    r"""Apply the 2d depthwise convolution.
    `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H, W)`, filters shape
      :math:`(C_{\text{out}}, 1, H_{\text{f}}, W_{\text{f}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, C_{\text{in}})`, filters shape
      :math:`(C_{\text{out}}, H_{\text{f}}, W_{\text{f}}, 1)`,
      and output shape is :math:`(N, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    ```python
    x = tf.ones((1, 2, 2, 2))
    filters = tf.ones((2, 1, 1, 1))
    y = tf.nn.depthwise_conv2d(x, filters)
    assert y.shape == (1, 2, 2, 2)
    ```

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The filters tensor.
    strides : Union[int, Sequence[int]]
        The stride of convolution window.
    padding : Union[int, Sequence[int], str]
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated filters.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return convolution(conv_type='depthwise_conv', **locals())


def dropout(x, rate, name=None, **kwargs):
    r"""Set the elements of input to zero randomly.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    The **Dropout** function is defined as:

    .. math:: \text{Dropout}(x) = x * \text{Bernoulli}(p=1 - prob)

    Examples:

    ```python
    x = tf.ones((2, 3), 'float32')
    print(tf.nn.dropout(x, 0.5))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The tensor :math:`x`.
    rate : Union[float, dragon.Tensor]
        The probability to zero an element.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.dropout(x, rate, name=name, **kwargs)


def elu(features, alpha=1., name=None, **kwargs):
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
    x = tf.constant([-1., 0., 1.])
    print(tf.nn.elu(x))
    ```

    Parameters
    ----------
    features : dragon.Tensor
        The tensor :math:`x`.
    alpha : float, optional, default=1.
        The value to :math:`\alpha`.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.elu(features, alpha=alpha, name=name, **kwargs)


def fused_batch_norm(
    x,
    scale,
    offset,
    mean,
    variance,
    epsilon=0.001,
    data_format='NHWC',
    is_training=True,
    name=None,
    exponential_avg_factor=1.0,
):
    r"""Apply the batch normalization.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    The moving average of stats are calculated as:

    .. math:: x_{\text{moving}} = \text{momentum} * x_{\text{moving}} +
                                  + (1 - \text{momentum}) * x_{\text{batch}}

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    scale : dragon.Tensor
        The :math:`\gamma` tensor.
    offset : dragon.Tensor
        The :math:`\beta` tensor.
    mean : dragon.Tensor
        The running mean tensor.
    variance : dragon.Tensor
        The running variance tensor.
    epsilon : float, optional, default=1e-3
        The value to :math:`\epsilon`.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    is_training : bool, optional, default=True
        The value to indicate training or inference.
    name : str, optional
        The operation name.
    exponential_avg_factor : float, optional, default=1.0
        The value to :math:`1 - \text{momentum}`.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return normalization_ops.batch_norm([
        x,
        scale,
        offset,
        mean,
        variance],
        axis=1 if data_format.startswith('NC') else -1,
        momentum=1 - exponential_avg_factor,
        epsilon=epsilon,
        use_stats=not is_training,
        name=name,
    )


def gelu(features, approximate=False, name=None):
    r"""Apply the gaussian error linear unit.
    `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

    The **GELU** function is defined as:

    .. math:: \text{GELU}(x) = x\cdot\frac{1}{2}[1 + \text{erf}(x / \sqrt{2})]

    Examples:

    ```python
    x = tf.constant([-1., 0., 1.])
    print(tf.nn.gelu(x))
    ```

    Parameters
    ----------
    features : dragon.Tensor
        The input tensor.
    approximate : bool, optional, default=False
        Whether to approximate the computation.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.gelu(features, approximate=approximate, name=name)


def l2_loss(t, name=None):
    r"""Compute the loss of element-wise squared error.

    The **L2Loss** function is defined as:

    .. math:: \text{L2Loss}(t) = sum(0.5 * t^{2})

    Examples:

    ```python
    t = tf.constant([-1., 2., -3.])
    print(tf.nn.l2_loss(t))  # 7.0
    ```

    Parameters
    ----------
    t : dragon.Tensor
        The input tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return loss_ops.l2_loss(t, reduction='sum', name=name) * 0.5


def l2_normalize(x, axis=None, epsilon=1e-12, name=None):
    r"""Apply the l2 normalization.

    The **L2-Normalization** is defined as:

    .. math:: y = \frac{x}{\left\|x\right\|_{2} + \epsilon}

    The argument ``axis`` could be negative or **None**:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]], 'float32')

    # A negative ``axis`` is the last-k axis
    print(tf.math.l2_normalize(x, 1))
    print(tf.math.l2_normalize(x, -1))  # Equivalent

    # If ``axis`` is None, the vector-style reduction
    # will be applied to compute a norm scalar
    print(tf.math.l2_normalize(x))

    # Also, ``axis`` could be a sequence of integers
    print(tf.math.l2_normalize(x, [0, 1]))
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The tensor :math:`x`.
    axis : Union[int, Sequence[int]], optional
        The axis to compute norm.
    epsilon : float, optional, default=1e-12
        The value to :math:`\epsilon`.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return normalization_ops.lp_norm(
        x, p=2, axis=axis, epsilon=epsilon, name=name)


def leaky_relu(features, alpha=0.2, name=None, **kwargs):
    r"""Apply the leaky rectified linear unit.

    The **LeakyReLU** function is defined as:

    .. math::
        \text{LeakyReLU}(x) =
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                \alpha * x, & \text{ otherwise }
            \end{cases}

    Parameters
    ----------
    features : dragon.Tensor
        The input tensor.
    alpha : number, optional, default=0.2
        The value to :math:`\alpha`.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.leaky_relu(features, alpha=alpha, name=name, **kwargs)


def local_response_normalization(
    input,
    depth_radius=5,
    bias=1.,
    alpha=1.,
    beta=0.5,
    data_format='NHWC',
    name=None,
):
    r"""Apply the local response normalization.
    `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

    The normalization is defined as:

    .. math::
        out_{i} = x_{i}\left(k + \frac{\alpha}{n}
            \sum_{j=\max(0, i-n/2)}^{\min(N-1,i+n/2)}x_{j}^2
        \right)^{-\beta}

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    depth_radius : int, optional, default=5
        The number of neighbouring channels to sum over.
    bias : float, optional, default=1.
        The bias constant :math:`k`.
    alpha : float, optional, default=1.
        The scale value :math:`\alpha`.
    beta : float, optional, default=0.5
        The exponent value :math:`\beta`.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return normalization_ops.local_response_norm(
        input,
        size=depth_radius,
        alpha=alpha,
        beta=beta,
        bias=bias,
        data_format=data_format,
        name=name,
    )


def log_softmax(logits, axis=-1, name=None):
    r"""Apply the composite of logarithm and softmax.

    The **LogSoftmax** function is defined as:

    .. math:: \text{LogSoftmax}(x) = \log(\frac{\exp(x_{i})}{\sum \exp(x_{j})})

    The argument ``axis`` could be negative:

    ```python
    x = tf.random.uniform((2, 3), -0.1, 0.1)
    print(tf.nn.log_softmax(x, 1))
    print(tf.nn.log_softmax(x, -1))  # Equivalent
    ```

    Parameters
    ----------
    logits : dragon.Tensor
        The input tensor.
    axis : int, optional, default=1
        The axis to reduce.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.log_softmax(logits, axis, name=name)


def max_pool(
    input,
    ksize,
    strides,
    padding='VALID',
    data_format='NHWC',
    name=None,
):
    r"""Apply the n-dimension max pooling.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, D1, D2, ...)`, and output shape is
      :math:`(N, C, D1_{\text{out}}, D2_{\text{out}}, ...)`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D1, D2, ..., C)`, and output shape is
      :math:`(N, D1_{\text{out}}, D2_{\text{out}}, ..., C)`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    for i in range(3):
        ndim = i + 1
        x = tf.ones((1,) + (2,) * ndim + (2,))
        y = tf.nn.max_pool(x, ksize=(2,) * ndim, strides=2)
        assert y.shape == (1,) + (1,) * ndim + (2,)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Union[int, Sequence[int]]
        The size of pooling window.
    strides : Union[int, Sequence[int]]
        The stride of pooling window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if input.shape is not None:
        num_total_dims = len(input.shape)
    else:
        num_total_dims = len(ksize)
    num_spatial_dims = num_total_dims - 2
    data_format = data_format if data_format else 'NHWC'
    start_axis = 2 if data_format.startswith('NC') else 1
    normalize_spatial_args = functools.partial(
        _normalize_spatial_args,
        num_total_dims=num_total_dims,
        num_spatial_dims=num_spatial_dims,
        start_axis=start_axis)
    ksize = normalize_spatial_args('ksize', ksize)
    strides = normalize_spatial_args('strides', strides)
    padding, pads = normalize_spatial_args('padding', padding)
    return getattr(vision_ops, 'pool{}d'.format(num_spatial_dims))(
        input,
        kernel_shape=ksize[start_axis:start_axis + num_spatial_dims],
        strides=strides[start_axis:start_axis + num_spatial_dims],
        padding=padding,
        pads=pads,
        mode='max',
        data_format='NCHW' if data_format.startswith('NC') else 'NHWC',
        name=name,
    )


def max_pool1d(
    input,
    ksize,
    strides,
    padding='VALID',
    data_format='NHWC',
    name=None,
):
    r"""Apply the 1d max pooling.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, H)`, and output shape is
      :math:`(N, C, H_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, C)`, and output shape is
      :math:`(N, H_{\text{out}}, C)`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2))
    y = tf.nn.max_pool1d(x, ksize=2, strides=2)
    assert y.shape == (1, 1, 2)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Union[int, Sequence[int]]
        The size of pooling window.
    strides : Union[int, Sequence[int]]
        The stride of pooling window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    start_axis = 2 if data_format.startswith('NC') else 1
    return max_pool(
        input=input,
        ksize=_normalize_spatial_args('ksize', ksize, 3, 1, start_axis),
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )


def max_pool2d(
    input,
    ksize,
    strides,
    padding='VALID',
    data_format='NHWC',
    name=None,
):
    r"""Apply the 2d max pooling.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, H, W)`, and output shape is
      :math:`(N, C, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, C)`, and output shape is
      :math:`(N, H_{\text{out}}, W_{\text{out}}, C)`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2, 2))
    y = tf.nn.max_pool2d(x, ksize=2, strides=2)
    assert y.shape == (1, 1, 1, 2)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Union[int, Sequence[int]]
        The size of pooling window.
    strides : Union[int, Sequence[int]], optional
        The stride of pooling window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    start_axis = 2 if data_format.startswith('NC') else 1
    return max_pool(
        input=input,
        ksize=_normalize_spatial_args('ksize', ksize, 4, 2, start_axis),
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )


def max_pool3d(
    input,
    ksize,
    strides,
    padding='VALID',
    data_format='NHWC',
    name=None,
):
    r"""Apply the 3d max pooling.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, D, H, W)`, and output shape is
      :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D, H, W, C)`, and output shape is
      :math:`(N, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}, C)`.

    * :attr:`padding` could be ``'VALID'``, ``'SAME'`` or explicit padding size.

    Examples:

    ```python
    x = tf.ones((1, 2, 2, 2, 2))
    y = tf.nn.max_pool3d(x, ksize=2, strides=2)
    assert y.shape == (1, 1, 1, 1, 2)
    ```

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Union[int, Sequence[int]]
        The size of pooling window.
    strides : Union[int, Sequence[int]]
        The stride of pooling window.
    padding : Union[int, Sequence[int], str], optional, default='VALID'
        The padding algorithm or size.
    data_format : str, optional, default='NHWC'
        ``'NCHW'`` or ``'NHWC'``.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    start_axis = 2 if data_format.startswith('NC') else 1
    return max_pool(
        input=input,
        ksize=_normalize_spatial_args('ksize', ksize, 5, 3, start_axis),
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )


def moments(x, axes=None, keepdims=False, name=None):
    r"""Compute the mean and variance of input along the given axis.

    .. math::
        \begin{cases}
            \mathrm{E}[x] = \frac{1}{n}\sum(x) \\
            \mathrm{Var}[x] = \frac{1}{n}\sum(x - \mathrm{E}[x])^{2}
        \end{cases}

    :attr:`axes` could be negative or ``None``:

    ```python
    x = tf.constant([[1, 2, 3], [4, 5, 6]])

    # A negative axis is the last-k axis
    print(tf.nn.moments(x, 1))
    print(tf.nn.moments(x, -1))  # Equivalent

    # If axes is None, reduce as a vector and return scalars
    print(tf.nn.moments(x))  # mean is 3.5, var is 2.916667

    # Also, axes could be a sequence of integers
    print(tf.nn.moments(x, [0, 1]))  # mean is 3.5, var is 2.916667
    ```

    Parameters
    ----------
    x : dragon.Tensor
        The input tensor.
    axes : Union[int, Sequence[int]], optional
        The axis to reduce.
    keepdims : bool, optional, default=False
        Keep the reduced dimensions or not.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The mean tensor.
    dragon.Tensor
        The variance tensor.

    """
    return math_ops.moments(x, axis=axes, keepdims=keepdims, name=name)


def relu(features, name=None, **kwargs):
    r"""Apply the rectified linear unit.
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
    features : dragon.Tensor
        The input tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.relu(features, name=name, **kwargs)


def relu6(features, name=None, **kwargs):
    r"""Apply the clipped-6 rectified linear unit.
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
    features : dragon.Tensor
        The input tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.relu6(features, name=name, **kwargs)


def selu(features, name=None, **kwargs):
    r"""Apply the scaled exponential linear unit.
    `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    .. math::
        \text{SELU}(x) = 1.0507 *
            \begin{cases}
                x, & \text{ if } x \geq 0 \\
                1.67326 * (\exp(x) - 1), & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = tf.constant([-1, 0, 1], 'float32')
    print(tf.nn.selu(x, inplace=False))
    ```

    Parameters
    ----------
    features : dragon.Tensor
        The tensor :math:`x`.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.selu(features, name=name, **kwargs)


def sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None):
    """Compute the loss of sigmoid cross entropy.

    Examples:

    ```python
    x = tf.constant([0.1, 0.2])
    y = tf.constant([0., 1.])
    print(tf.nn.sigmoid_cross_entropy_with_logits(y, x))  # 0.744, 0.598
    ```

    Parameters
    ----------
    labels : dragon.Tensor
        The target tensor.
    logits : dragon.Tensor
        The input tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return loss_ops.sigmoid_cross_entropy_loss(
        [logits, labels], reduction='none', name=name)


def silu(features):
    r"""Apply the sigmoid linear unit.
    `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

    The **SiLU** function is defined as:

    .. math:: \text{SiLU}(x) = x \cdot \frac{1}{1 + \exp(-x)}

    Examples:

    ```python
    x = tf.constant([-2.5, -1.0, 0.0, 1.0, 2.5])
    print(tf.nn.silu(x))
    ```

    Parameters
    ----------
    features : dragon.Tensor
        The input tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.silu(features)


def softmax(logits, axis=-1, name=None, **kwargs):
    r"""Apply the softmax function.

    The **Softmax** function is defined as:

    .. math:: \text{Softmax}(x_{i}) = \frac{\exp(x_{i})}{\sum_{j} \exp(x_{j})}

    The argument ``axis`` could be negative:

    ```python
    x = tf.ones((1, 4), dtype='float32')
    print(tf.nn.softmax(x, 1))   # [[0.25 0.25 0.25 0.25]]
    print(tf.nn.softmax(x, -1))  # Equivalent
    ```

    Parameters
    ----------
    logits : dragon.Tensor
        The input tensor.
    axis : int, optional, default=-1
        The axis to reduce.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return activation_ops.softmax(logits, axis=axis, name=name, **kwargs)


def softmax_cross_entropy_with_logits(labels, logits, name=None):
    """Compute the loss of softmax cross entropy.

    Examples:

    ```python
    labels = tf.constant([[0., 1., ], [1., 0.]], dtype=tf.float32)
    logits = tf.constant([[0.5, 0.5], [0.3, 0.7]], dtype=tf.float32)
    print(tf.nn.softmax_cross_entropy_with_logits(labels, logits))  # [0.6931472, 0.9130153]
    ```

    Parameters
    ----------
    labels : dragon.Tensor
        The label tensor.
    logits : dragon.Tensor
        The logit tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return loss_ops.softmax_cross_entropy_loss(
        [logits, labels], axis=-1, reduction='none', name=name)


def sparse_softmax_cross_entropy_with_logits(labels, logits, name=None):
    """Compute the loss of softmax cross entropy with sparse labels.

    Examples:

    ```python
    labels = tf.constant([1, 0], dtype=tf.int64)
    logits = tf.constant([[0.5, 0.5], [0.3, 0.7]], dtype=tf.float32)
    print(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))  # [0.6931472, 0.9130153]
    ```

    Parameters
    ----------
    labels : dragon.Tensor
        The label tensor.
    logits : dragon.Tensor
        The logit tensor.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return loss_ops.softmax_cross_entropy_loss(
        [logits, labels], axis=-1, reduction='none', name=name)


def top_k(input, k=1, sorted=True, name=None):
    """Return the top k-largest elements along the last axis.

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    k : int, optional, default=1
        The number of top elements to select.
    sorted : bool, optional
        Whether to return the elements in the sorted order.
    name : str, optional
        The operation name.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The value and index tensor.

    """
    return sort_ops.top_k(input, k, sorted=sorted, name=name)


def _normalize_spatial_args(
    name,
    values,
    num_total_dims,
    num_spatial_dims,
    start_axis,
):
    if name in ('ksize', 'strides', 'dilations'):
        if values is None:
            return [1] * num_total_dims
        else:
            values = nest.flatten(values)
            if len(values) != num_total_dims:
                defaults, n_provides = [1] * num_total_dims, len(values)
                if n_provides != num_spatial_dims:
                    if n_provides == 1:
                        values = values * num_spatial_dims
                    else:
                        raise ValueError(
                            'Except 1, {} or {} values for <{}>.'
                            .format(num_spatial_dims, num_spatial_dims * 2, name))
                defaults[start_axis:start_axis + len(values)] = values
                return defaults
            return values
    elif name == 'padding':
        if isinstance(values, six.string_types):
            padding, pads = values.upper(), 0
        else:
            padding_tuple = nest.flatten(values)
            padding = 'VALID'
            if len(padding_tuple) == 1:
                pads = padding_tuple[0]
            elif len(padding_tuple) == num_spatial_dims:
                pads = padding_tuple
            elif len(padding_tuple) == (num_spatial_dims * 2):
                pads_l, pads_r = [], []
                for i in range(num_spatial_dims):
                    pads_l.append(padding_tuple[i * 2])
                    pads_r.append(padding_tuple[i * 2 + 1])
                pads = pads_l + pads_r
            else:
                raise ValueError(
                    'Except 1, {} or {} values if <padding> set as explict pads.'
                    .format(num_spatial_dims, num_spatial_dims * 2))
        return padding, pads
    elif name == 'output_shape':
        if values is not None:
            if types.is_tensor(values):
                values = values.numpy().tolist()
            values = nest.flatten(values)
            if len(values) != num_spatial_dims:
                raise ValueError('Except {} values for <output_shape>.'
                                 .format(num_spatial_dims))
        return values
