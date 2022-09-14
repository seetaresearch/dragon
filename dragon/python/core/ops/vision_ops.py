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

from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema
from dragon.core.util import nest


@OpSchema.num_inputs(2)
def bias_add(inputs, data_format='NCHW', inplace=False, **kwargs):
    """Add the bias across channels to input.

    Examples:

    ```python
    x = dragon.ones((1, 2, 3, 3))
    bias = dragon.ones((2,))
    print(dragon.nn.bias_add([x, bias]))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The ``x`` and ``bias``.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.
    inplace : bool, optional, default=False
        Call in-place or return a new tensor.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    if context.executing_eagerly():
        return OpLib.execute(
            'BiasAdd', inputs, outputs=[inputs if inplace else None],
            data_format=data_format)
    return OpLib.add('BiasAdd', inputs, data_format=data_format, **kwargs)


@OpSchema.num_inputs(2, 3)
def conv(
    inputs,
    kernel_shape=(3, 3),
    strides=1,
    pads=0,
    dilations=1,
    group=1,
    padding='VALID',
    data_format='NCHW',
    **kwargs
):
    r"""Apply the n-dimension convolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, D1, D2, ...)`, weight shape
      :math:`(C_{\text{out}}, C_{\text{in}}, D1_{\text{k}}, D2_{\text{k}}, ...)`,
      and output shape is :math:`(N, C_{\text{out}}, D1_{\text{out}}, D2_{\text{out}}, ...)`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D1, D2, ..., C_{\text{in}})`, weight shape
      :math:`(C_{\text{out}}, D1_{\text{k}}, D2_{\text{k}}, ..., C_{\text{in}})`,
      and output shape is :math:`(N, D1_{\text{out}}, D2_{\text{out}}, ..., C_{\text{out}})`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    for i in range(3):
        ndim = i + 1
        x = dragon.ones((1, 2) + (2,) * ndim)
        w = dragon.ones((3, 2) + (1,) * ndim)
        y = dragon.nn.conv([x, w], kernel_shape=(1,) * ndim)
        assert y.shape == (1, 3) + (2,) * ndim
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and ``bias``.
    kernel_shape : Sequence[int], optional, default=(3, 3)
        The shape of convolution window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated convolution.
    group : int, optional, default=1
        The number of groups to split channels into.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: %s' % padding)
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    strides = nest.flatten(strides)
    pads = nest.flatten(pads)
    dilations = nest.flatten(dilations)
    if context.executing_eagerly():
        weight_shape = inputs[1].shape
        return OpLib.execute(
            'Conv',
            inputs,
            dim_in=weight_shape[1],
            dim_out=weight_shape[0],
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
            dilations=dilations,
            group=group,
            padding=padding,
            data_format=data_format,
            bias=len(inputs) > 2,
            dtype=inputs[1].dtype,
            input_shape=inputs[0].shape)
    return OpLib.add('Conv', inputs,
                     kernel_shape=kernel_shape,
                     strides=strides,
                     pads=pads,
                     dilations=dilations,
                     group=group,
                     padding=padding,
                     data_format=data_format,
                     **kwargs)


@OpSchema.num_inputs(2, 3)
def conv1d(
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
    r"""Apply the 1d convolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H)`, weight shape
      :math:`(C_{\text{out}}, C_{\text{in}}, H_{\text{k}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, C_{\text{in}})`, weight shape
      :math:`(C_{\text{out}}, H_{\text{k}}, C_{\text{in}})`,
      and output shape is :math:`(N, H_{\text{out}}, C_{\text{out}})`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    x = dragon.ones((1, 2, 2))
    w = dragon.ones((3, 2, 1))
    y = dragon.nn.conv1d([x, w], kernel_shape=1)
    assert y.shape == (1, 3, 2)
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and optional ``bias``.
    kernel_shape : Union[int, Sequence[int]], optional, default=3
        The shape of convolution window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated convolution.
    group : int, optional, default=1
        The number of groups to split channels into.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return conv(
        inputs,
        kernel_shape=_normalize_tuple(kernel_shape, 1),
        strides=_normalize_tuple(strides, 1),
        pads=_normalize_pads(pads, 1),
        dilations=_normalize_tuple(dilations, 1),
        group=group,
        padding=padding,
        data_format=data_format,
        **kwargs
    )


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

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H, W)`, weight shape
      :math:`(C_{\text{out}}, C_{\text{in}}, H_{\text{k}}, W_{\text{k}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, C_{\text{in}})`, weight shape
      :math:`(C_{\text{out}}, H_{\text{k}}, W_{\text{k}}, C_{\text{in}})`,
      and output shape is :math:`(N, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    x = dragon.ones((1, 2, 2, 2))
    w = dragon.ones((3, 2, 1, 1))
    y = dragon.nn.conv2d([x, w], kernel_shape=1)
    assert y.shape == (1, 3, 2, 2)
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and optional ``bias``.
    kernel_shape : Union[int, Sequence[int]], optional, default=3
        The shape of convolution window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated convolution.
    group : int, optional, default=1
        The number of groups to split channels into.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return conv(
        inputs,
        kernel_shape=_normalize_tuple(kernel_shape, 2),
        strides=_normalize_tuple(strides, 2),
        pads=_normalize_pads(pads, 2),
        dilations=_normalize_tuple(dilations, 2),
        group=group,
        padding=padding,
        data_format=data_format,
        **kwargs
    )


@OpSchema.num_inputs(2, 3)
def conv3d(
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
    r"""Apply the 3d convolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, D, H, W)`, weight shape
      :math:`(C_{\text{out}}, C_{\text{in}}, D_{\text{k}}, H_{\text{k}}, W_{\text{k}})`,
      and output shape is :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, D, C_{\text{in}})`, weight shape
      :math:`(C_{\text{out}}, D_{\text{k}}, H_{\text{k}}, W_{\text{k}}, C_{\text{in}})`,
      and output shape is :math:`(N, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    ```python
    x = dragon.ones((1, 2, 2, 2, 2))
    w = dragon.ones((3, 2, 1, 1, 1))
    y = dragon.nn.conv3d([x, w], kernel_shape=1)
    assert y.shape == (1, 3, 2, 2, 2)
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and optional ``bias``.
    kernel_shape : Union[int, Sequence[int]], optional, default=3
        The shape of convolution window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated convolution.
    group : int, optional, default=1
        The number of groups to split channels into.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return conv(
        inputs,
        kernel_shape=_normalize_tuple(kernel_shape, 3),
        strides=_normalize_tuple(strides, 3),
        pads=_normalize_pads(pads, 3),
        dilations=_normalize_tuple(dilations, 3),
        group=group,
        padding=padding,
        data_format=data_format,
        **kwargs
    )


@OpSchema.num_inputs(2, 3)
@OpSchema.convert_arg('output_padding')
@OpSchema.convert_arg('output_shape')
def conv_transpose(
    inputs,
    kernel_shape=(3, 3),
    strides=1,
    pads=0,
    dilations=1,
    group=1,
    padding='VALID',
    output_padding=None,
    output_shape=None,
    data_format='NCHW',
    **kwargs
):
    r"""Apply the n-dimension deconvolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, D1, D2, ...)`, weight shape
      :math:`(C_{\text{in}}, C_{\text{out}}, D1_{\text{k}}, D2_{\text{k}}, ...)`,
      and output shape is :math:`(N, C_{\text{out}}, D1_{\text{out}}, D2_{\text{out}}, ...)`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D1, D2, ..., C_{\text{in}})`, weight shape
      :math:`(C_{\text{in}}, D1_{\text{k}}, D2_{\text{k}}, ..., C_{\text{out}})`,
      and output shape is :math:`(N, D1_{\text{out}}, D2_{\text{out}}, ..., C_{\text{out}})`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    for i in range(3):
        ndim = i + 1
        x = dragon.ones((1, 2) + (2,) * ndim)
        w = dragon.ones((3, 2) + (1,) * ndim)
        y = dragon.nn.conv_transpose(
            [x, w],
            kernel_shape=(1,) * ndim,
            output_shape=(3,) * ndim,
            output_padding=(1,) * ndim)
        assert y.shape == (1, 3) + (3,) * ndim
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and optional ``bias``.
    kernel_shape : Sequence[int], optional, default=(3, 3)
        The shape of convolution window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated convolution.
    group : int, optional, default=1
        The number of groups to split channels into.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    output_padding : Union[Sequence[int], dragon.Tensor], optional
        The additional size added to the output shape.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The output shape for automatic padding.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: %s' % padding)
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    if 'SAME' in padding and output_shape is None:
        raise ValueError('Excepted <output_shape> for same padding.')
    if output_shape is not None and 'SAME' not in padding:
        args['padding'] = 'SAME'
    for k in ('strides', 'pads', 'dilations'):
        args[k] = nest.flatten(args[k])
    if context.executing_eagerly():
        weight_shape = inputs[1].shape
        return OpLib.execute(
            'ConvTranspose',
            inputs,
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
            input_shape=inputs[0].shape)
    return OpLib.add('ConvTranspose', **args)


@OpSchema.num_inputs(min_num=2, max_num=3)
def conv1d_transpose(
    inputs,
    kernel_shape=3,
    strides=1,
    pads=0,
    dilations=1,
    group=1,
    padding='VALID',
    output_padding=None,
    output_shape=None,
    data_format='NCHW',
    **kwargs
):
    r"""Apply the 1d deconvolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H)`, weight shape
      :math:`(C_{\text{in}}, C_{\text{out}}, H_{\text{k}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, C_{\text{in}})`, weight shape
      :math:`(C_{\text{in}}, H_{\text{k}}, C_{\text{out}})`,
      and output shape is :math:`(N, H_{\text{out}}, C_{\text{out}})`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    x = dragon.ones((1, 2, 2))
    w = dragon.ones((3, 2, 1))
    y = dragon.nn.conv1d_transpose(
        [x, w],
        kernel_shape=1,
        output_shape=(3,),
        output_padding=(1,))
    assert y.shape == (1, 3, 3)
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and optional ``bias``.
    kernel_shape : Union[int, Sequence[int]], optional, default=3
        The shape of convolution window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated convolution.
    group : int, optional, default=1
        The number of groups to split channels into.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    output_padding : Union[Sequence[int], dragon.Tensor], optional
        The additional size added to the output shape.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The output shape for automatic padding.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return conv_transpose(
        inputs,
        kernel_shape=_normalize_tuple(kernel_shape, 1),
        strides=_normalize_tuple(strides, 1),
        pads=_normalize_pads(pads, 1),
        dilations=_normalize_tuple(dilations, 1),
        group=group,
        padding=padding,
        output_padding=output_padding,
        output_shape=output_shape,
        data_format=data_format,
        **kwargs
    )


@OpSchema.num_inputs(min_num=2, max_num=3)
def conv2d_transpose(
    inputs,
    kernel_shape=3,
    strides=1,
    pads=0,
    dilations=1,
    group=1,
    padding='VALID',
    output_padding=None,
    output_shape=None,
    data_format='NCHW',
    **kwargs
):
    r"""Apply the 2d deconvolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H, W)`, weight shape
      :math:`(C_{\text{in}}, C_{\text{out}}, H_{\text{k}}, W_{\text{k}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, C_{\text{in}})`, weight shape
      :math:`(C_{\text{in}}, H_{\text{k}}, W_{\text{k}}, C_{\text{out}})`,
      and output shape is :math:`(N, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    x = dragon.ones((1, 2, 2, 2))
    w = dragon.ones((3, 2, 1, 1))
    y = dragon.nn.conv2d_transpose(
        [x, w],
        kernel_shape=1,
        output_shape=(3, 3),
        output_padding=(1, 1))
    assert y.shape == (1, 3, 3, 3)
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and optional ``bias``.
    kernel_shape : Union[int, Sequence[int]], optional, default=3
        The shape of convolution window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated convolution.
    group : int, optional, default=1
        The number of groups to split channels into.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    output_padding : Union[Sequence[int], dragon.Tensor], optional
        The additional size added to the output shape.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The output shape for automatic padding.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return conv_transpose(
        inputs,
        kernel_shape=_normalize_tuple(kernel_shape, 2),
        strides=_normalize_tuple(strides, 2),
        pads=_normalize_pads(pads, 2),
        dilations=_normalize_tuple(dilations, 2),
        group=group,
        padding=padding,
        output_padding=output_padding,
        output_shape=output_shape,
        data_format=data_format,
        **kwargs
    )


@OpSchema.num_inputs(min_num=2, max_num=3)
def conv3d_transpose(
    inputs,
    kernel_shape=3,
    strides=1,
    pads=0,
    dilations=1,
    group=1,
    padding='VALID',
    output_padding=None,
    output_shape=None,
    data_format='NCHW',
    **kwargs
):
    r"""Apply the 3d deconvolution.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, D, H, W)`, weight shape
      :math:`(C_{\text{in}}, C_{\text{out}}, D_{\text{k}}, H_{\text{k}}, W_{\text{k}})`,
      and output shape is :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, D, C_{\text{in}})`, weight shape
      :math:`(C_{\text{in}}, D_{\text{k}}, H_{\text{k}}, W_{\text{k}}, C_{\text{out}})`,
      and output shape is :math:`(N, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    x = dragon.ones((1, 2, 2, 2, 2))
    w = dragon.ones((3, 2, 1, 1, 1))
    y = dragon.nn.conv3d_transpose(
        [x, w],
        kernel_shape=1,
        output_shape=(3, 3, 3),
        output_padding=(1, 1, 1))
    assert y.shape == (1, 3, 3, 3, 3)
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and optional ``bias``.
    kernel_shape : Union[int, Sequence[int]], optional, default=3
        The shape of convolution window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated convolution.
    group : int, optional, default=1
        The number of groups to split channels into.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    output_padding : Union[Sequence[int], dragon.Tensor], optional
        The additional size added to the output shape.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The output shape for automatic padding.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return conv_transpose(
        inputs,
        kernel_shape=_normalize_tuple(kernel_shape, 3),
        strides=_normalize_tuple(strides, 3),
        pads=_normalize_pads(pads, 3),
        dilations=_normalize_tuple(dilations, 3),
        group=group,
        padding=padding,
        output_padding=output_padding,
        output_shape=output_shape,
        data_format=data_format,
        **kwargs
    )


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

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C_{\text{in}}, H, W)`, weight shape
      :math:`(C_{\text{out}}, 1, H_{\text{k}}, W_{\text{k}})`,
      and output shape is :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, C_{\text{in}})`, weight shape
      :math:`(C_{\text{out}}, H_{\text{k}}, W_{\text{k}}, 1)`,
      and output shape is :math:`(N, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    x = dragon.ones((1, 2, 2, 2))
    w = dragon.ones((2, 1, 1, 1))
    y = dragon.nn.depthwise_conv2d([x, w], kernel_shape=1)
    assert y.shape == (1, 2, 2, 2)
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x``, ``weight`` and optional ``bias``.
    kernel_shape : Union[int, Sequence[int]], optional, default=3
        The shape of convolution window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of convolution window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The rate of dilated convolution.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: %s' % padding)
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    kernel_shape = _normalize_tuple(kernel_shape, 2)
    strides = _normalize_tuple(strides, 2)
    pads = _normalize_pads(pads, 2)
    dilations = _normalize_tuple(dilations, 2)
    if context.executing_eagerly():
        weight_shape = inputs[1].shape
        return OpLib.execute(
            'DepthwiseConv',
            inputs,
            dim_in=weight_shape[1],
            dim_out=weight_shape[0],
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
            dilations=dilations,
            padding=padding,
            data_format=data_format,
            bias=len(inputs) > 2,
            dtype=inputs[1].dtype)
    return OpLib.add('DepthwiseConv', inputs,
                     kernel_shape=kernel_shape,
                     strides=strides,
                     pads=pads,
                     dilations=dilations,
                     padding=padding,
                     data_format=data_format,
                     **kwargs)


@OpSchema.num_inputs(1)
def depth_to_space(
    inputs,
    block_size,
    mode='DCR',
    data_format='NCHW',
    copy=True,
    **kwargs
):
    """Rearrange depth data into spatial blocks.

    Examples:

    ```python
    n, c, h, w, bs = 1, 4, 1, 1, 2
    x = dragon.range(n * c * h * w).reshape((n, c, h, w))
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
    mode : str, optional, default='DCR'
        Rearrangement order for ``'NCHW'`` format.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    if context.executing_eagerly():
        return OpLib.execute(
            'DepthToSpace', inputs,
            outputs=[None] if copy else inputs,
            block_size=block_size, mode=mode.upper(), data_format=data_format)
    return OpLib.add('DepthToSpace', inputs, block_size=block_size,
                     mode=mode.upper(), data_format=data_format, **kwargs)


@OpSchema.num_inputs(1)
def extract_patches(
    inputs,
    kernel_shape=(3, 3),
    strides=1,
    pads=0,
    dilations=1,
    padding='VALID',
    data_format='NCHW',
    **kwargs
):
    r"""Extract the sliding patches from input.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, D1, D2, ...)`, and output shape is
      :math:`(N, C \times \prod(\text{kernel\_shape}), D1_{\text{out}}, D2_{\text{out}}, ...)`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D1, D2, ..., C)`, and output shape is
      :math:`(N, D1_{\text{out}}, D2_{\text{out}}, ..., \prod(\text{kernel\_shape}) \times C)`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    for i in range(3):
        ndim = i + 1
        x = dragon.ones((1, 2) + (2,) * ndim)
        y = dragon.vision.extract_patches(x, kernel_shape=(2,) * ndim)
        assert y.shape == (1, 2 * (2 ** ndim)) + (1,) * ndim
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    kernel_shape : Sequence[int], optional, default=(3, 3)
        The shape of sliding window.
    strides : Union[int, Sequence[int]], optional, default=1
        The stride of sliding window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    dilations : Union[int, Sequence[int]], optional, default=1
        The dilated rate of sliding window.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: %s' % padding)
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    kernel_shape = nest.flatten(kernel_shape)
    if context.executing_eagerly():
        return OpLib.execute(
            'Im2Col',
            inputs,
            kernel_shape=kernel_shape,
            strides=_normalize_tuple(strides, len(kernel_shape)),
            pads=_normalize_pads(pads, len(kernel_shape)),
            dilations=_normalize_tuple(dilations, len(kernel_shape)),
            padding=padding,
            data_format=data_format,
        )
    return OpLib.add('Im2Col', inputs,
                     kernel_shape=kernel_shape,
                     strides=_normalize_tuple(strides, len(kernel_shape)),
                     pads=_normalize_pads(pads, len(kernel_shape)),
                     dilations=_normalize_tuple(dilations, len(kernel_shape)),
                     padding=padding,
                     data_format=data_format,
                     **kwargs)


@OpSchema.num_inputs(1)
def pool(
    inputs,
    kernel_shape,
    strides,
    pads=0,
    padding='VALID',
    mode='max',
    global_pool=False,
    ceil_mode=False,
    data_format='NCHW',
    **kwargs
):
    r"""Apply the n-dimension pooling.

    * Set :attr:`mode` for the specific pooling type, default is ``maxpool``.

    * Use :attr:`global_pool` to apply the global pooling further.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, D1, D2, ...)`, and output shape is
      :math:`(N, C, D1_{\text{out}}, D2_{\text{out}}, ...)`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D1, D2, ..., C)`, and output shape is
      :math:`(N, D1_{\text{out}}, D2_{\text{out}}, ..., C)`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    for i in range(3):
        ndim = i + 1
        x = dragon.ones((1, 2) + (2,) * ndim)
        y = dragon.nn.pool(x, kernel_shape=(2,) * ndim, strides=2)
        assert y.shape == (1, 2) + (1,) * ndim
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    kernel_shape : Sequence[int], required
        The shape of pooling window.
    strides : Union[int, Sequence[int]], required
        The stride of pooling window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    mode : str, optional, default='max'
        ``'max'`` or ``'avg'``.
    global_pool : bool, optional, default=False
        Apply the global pooling or not.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    mode = mode.upper()
    if mode not in ('MAX', 'AVG'):
        raise ValueError('Unsupported pooling mode: %s' % mode)
    if padding not in ('VALID', 'SAME', 'SAME_UPPER', 'SAME_LOWER'):
        raise ValueError('Unsupported padding algorithm: %s' % padding)
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    strides = nest.flatten(strides)
    pads = nest.flatten(pads)
    if context.executing_eagerly():
        return OpLib.execute(
            'Pool',
            inputs,
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
            padding=padding,
            ceil_mode=ceil_mode,
            mode=mode,
            data_format=data_format,
            global_pool=global_pool)
    return OpLib.add('Pool', inputs,
                     kernel_shape=kernel_shape,
                     strides=strides,
                     pads=pads,
                     padding=padding,
                     ceil_mode=ceil_mode,
                     mode=mode,
                     data_format=data_format,
                     global_pool=global_pool,
                     **kwargs)


@OpSchema.num_inputs(1)
def pool1d(
    inputs,
    kernel_shape,
    strides,
    pads=0,
    padding='VALID',
    mode='max',
    global_pool=False,
    ceil_mode=False,
    data_format='NCHW',
    **kwargs
):
    r"""Apply the 1d pooling.

    * Set :attr:`mode` for the specific pooling type, default is ``maxpool``.

    * Use :attr:`global_pool` to apply the global pooling further.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, H)`, and output shape is :math:`(N, C, H_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, C)`, and output shape is :math:`(N, H_{\text{out}}, C)`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    x = dragon.ones((1, 2, 2))
    y = dragon.nn.pool1d(x, kernel_shape=2, strides=2)
    assert y.shape == (1, 2, 1)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    kernel_shape : Union[int, Sequence[int]], required
        The shape of pooling window.
    strides : Union[int, Sequence[int]], required
        The stride of pooling window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    mode : str, optional, default='max'
        ``'max'`` or ``'avg'``.
    global_pool : bool, optional, default=False
        Apply the global pooling or not.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return pool(
        inputs,
        kernel_shape=_normalize_tuple(kernel_shape, 1),
        strides=_normalize_tuple(strides, 1),
        pads=_normalize_pads(pads, 1),
        padding=padding,
        ceil_mode=ceil_mode,
        mode=mode,
        data_format=data_format,
        global_pool=global_pool,
        **kwargs)


@OpSchema.num_inputs(1)
def pool2d(
    inputs,
    kernel_shape,
    strides,
    pads=0,
    padding='VALID',
    mode='max',
    global_pool=False,
    ceil_mode=False,
    data_format='NCHW',
    **kwargs
):
    r"""Apply the 2d pooling.

    * Set :attr:`mode` for the specific pooling type, default is ``maxpool``.

    * Use :attr:`global_pool` to apply the global pooling further.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, H, W)`, and output shape is
      :math:`(N, C, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, H, W, C)`, and output shape is
      :math:`(N, H_{\text{out}}, W_{\text{out}}, C)`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    x = dragon.ones((1, 2, 2, 2))
    y = dragon.nn.pool2d(x, kernel_shape=2, strides=2)
    assert y.shape == (1, 2, 1, 1)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    kernel_shape : Union[int, Sequence[int]], required
        The shape of pooling window.
    strides : Union[int, Sequence[int]], required
        The stride of pooling window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    mode : str, optional, default='max'
        ``'max'`` or ``'avg'``.
    global_pool : bool, optional, default=False
        Apply the global pooling or not.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return pool(
        inputs,
        kernel_shape=_normalize_tuple(kernel_shape, 2),
        strides=_normalize_tuple(strides, 2),
        pads=_normalize_pads(pads, 2),
        padding=padding,
        ceil_mode=ceil_mode,
        mode=mode,
        data_format=data_format,
        global_pool=global_pool,
        **kwargs)


@OpSchema.num_inputs(1)
def pool3d(
    inputs,
    kernel_shape,
    strides,
    pads=0,
    padding='VALID',
    mode='max',
    global_pool=False,
    ceil_mode=False,
    data_format='NCHW',
    **kwargs
):
    r"""Apply the 3d pooling.

    * Set :attr:`mode` for the specific pooling type, default is ``maxpool``.

    * Use :attr:`global_pool` to apply the global pooling further.

    * If :attr:`data_format` is ``'NCHW'``, excepts input shape
      :math:`(N, C, D, H, W)`, and output shape is
      :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    * If :attr:`data_format` is ``'NHWC'``, excepts input shape
      :math:`(N, D, H, W, C)`, and output shape is
      :math:`(N, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}, C)`.

    * If :attr:`padding` is ``'VALID'``, :attr:`pads` controls the explicit padding size.
      Otherwise, size are computed automatically use the given method.

    Examples:

    ```python
    x = dragon.ones((1, 2, 2, 2, 2))
    y = dragon.nn.pool3d(x, kernel_shape=2, strides=2)
    assert y.shape == (1, 2, 1, 1, 1)
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    kernel_shape : Union[int, Sequence[int]], required
        The shape of pooling window.
    strides : Union[int, Sequence[int]], required
        The stride of pooling window.
    pads : Union[int, Sequence[int]], optional, default=0
        The zero padding size.
    padding : str, optional, default='VALID'
        ``'VALID'``, ``'SAME'``, ``'SAME_UPPER'`` or ``'SAME_LOWER'``.
    mode : str, optional, default='max'
        ``'max'`` or ``'avg'``.
    global_pool : bool, optional, default=False
        Apply the global pooling or not.
    ceil_mode : bool, optional, default=False
        Ceil or floor the boundary.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return pool(
        inputs,
        kernel_shape=_normalize_tuple(kernel_shape, 3),
        strides=_normalize_tuple(strides, 3),
        pads=_normalize_pads(pads, 3),
        padding=padding,
        ceil_mode=ceil_mode,
        mode=mode,
        data_format=data_format,
        global_pool=global_pool,
        **kwargs)


@OpSchema.num_inputs(1)
@OpSchema.convert_arg('sizes')
@OpSchema.convert_arg('scales')
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

    :attr:`sizes` or :attr:`scales` will be selected by :attr:`data_format`:

    ```python
    x, sizes = dragon.ones((1, 2, 3, 4)), (6, 6)
    a = dragon.vision.resize(x, sizes, data_format='NCHW')  # Shape: (1, 2, 6, 6)
    c = dragon.vision.resize(x, sizes, data_format='NHWC')  # Shape: (1, 6, 6, 4)
    ```

    Use :attr:`align_corners` to determine the input coordinates in linear interpolating:

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
    mode : str, optional, default='nearest'
        ``'nearest'`` or ``'linear'``.
    align_corners : bool, optional, default=False
        Whether to align corners in linear interpolating.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    args = OpSchema.parse_args(locals())
    args['mode'] = mode.upper()
    if sizes is None and scales is None:
        raise ValueError('Specify either <sizes> or <scales>.')
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))
    if context.executing_eagerly():
        if sizes is not None:
            args['sizes'] = nest.flatten(args['sizes'])
        if scales is not None:
            args['scales'] = nest.flatten(args['scales'])
        return OpLib.execute(
            'Resize',
            inputs,
            mode=args['mode'],
            align_corners=align_corners,
            num_sizes=len(args['sizes']) if sizes is not None else 0,
            num_scales=len(args['scales']) if scales is not None else 0,
            data_format=data_format,
            sizes=args['sizes'],
            scales=args['scales'])
    return OpLib.add('Resize', **args)


@OpSchema.num_inputs(2)
def roi_align(
    inputs,
    pooled_h,
    pooled_w,
    spatial_scale=1.0,
    sampling_ratio=-1,
    aligned=False,
    **kwargs
):
    r"""Apply the average roi align.
    `[He et.al, 2017] <https://arxiv.org/abs/1703.06870>`_.

    The input ``rois`` should be packed with the shape :math:`(N, 5)`,
    where :math:`N` is the number of RoIs, and each column takes :math:`5` values
    for a sequence of :math:`[i_{\text{batch}}, x_{\min}, y_{\min}, x_{\max}, y_{\max}]`.

    Examples:

    ```python
    x = dragon.range(18, dtype='float32').reshape((1, 2, 3, 3))
    rois = dragon.constant([[0., 1., 1., 2.]], dtype='float32')
    print(dragon.vision.roi_align([x, rois], pooled_h=1, pooled_w=1))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x`` and ``rois``.
    pooled_h : int, required
        The output height.
    pooled_w : int, required
        The output width.
    spatial_scale : float, optional, default=1.0
        The input scale to the size of ``rois``.
    sampling_ratio : int, optional, default=-1
        The number of sampling grids for ``rois``.
    aligned : bool, optional, default=False
        Whether to shift the input coordinates by ``-0.5``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    spatial_scale = float(spatial_scale)
    if context.executing_eagerly():
        return OpLib.execute(
            'RoiAlign',
            inputs,
            pooled_h=pooled_h,
            pooled_w=pooled_w,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned)
    return OpLib.add('RoiAlign', inputs,
                     pooled_h=pooled_h,
                     pooled_w=pooled_w,
                     spatial_scale=spatial_scale,
                     sampling_ratio=sampling_ratio,
                     aligned=aligned,
                     **kwargs)


@OpSchema.num_inputs(2)
def roi_pool(
    inputs,
    pooled_h,
    pooled_w,
    spatial_scale=1.0,
    **kwargs
):
    r"""Apply the max roi pooling.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    The input ``rois`` should be packed with the shape :math:`(N, 5)`,
    where :math:`N` is the number of RoIs, and each column takes :math:`5` values
    for a sequence of :math:`[i_{\text{batch}}, x_{\min}, y_{\min}, x_{\max}, y_{\max}]`.

    Examples:

    ```python
    x = dragon.range(18, dtype='float32').reshape((1, 2, 3, 3))
    rois = dragon.constant([[0., 1., 1., 2.]], dtype='float32')
    print(dragon.vision.roi_pool([x, rois], pooled_h=1, pooled_w=1))
    ```

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The tensor ``x`` and ``rois``.
    pooled_h : int, required
        The output height.
    pooled_w : int, required
        The output width.
    spatial_scale : float, optional, default=1.0
        The input scale to the size of ``rois``.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    spatial_scale = float(spatial_scale)
    if context.executing_eagerly():
        return OpLib.execute(
            'RoiPool',
            inputs,
            pooled_h=pooled_h,
            pooled_w=pooled_w,
            spatial_scale=spatial_scale)
    return OpLib.add('RoiPool', inputs,
                     pooled_h=pooled_h,
                     pooled_w=pooled_w,
                     spatial_scale=spatial_scale,
                     **kwargs)


@OpSchema.num_inputs(1)
def space_to_depth(
    inputs,
    block_size,
    mode='DCR',
    data_format='NCHW',
    copy=True,
    **kwargs
):
    """Rearrange blocks of spatial data into depth.

    Examples:

    ```python
    n, c, h, w, bs = 1, 2, 2, 2, 2
    x = dragon.range(n * c * h * w).reshape((n, c, h, w))
    y = dragon.reshape(x, (n, c, h // bs, bs, w // bs, bs))
    y = dragon.transpose(y, (0, 3, 5, 1, 2, 4))
    y = dragon.reshape(y, (n, (bs ** 2) * c, h // bs, w // bs))
    z = dragon.nn.space_to_depth(x, 2)  # Equivalent
    ```

    Parameters
    ----------
    inputs : dragon.Tensor
        The input tensor.
    block_size : int, required
        The size of spatial block.
    mode : str, optional, default='DCR'
        Rearrangement order for ``'NCHW'`` format.
    data_format : str, optional, default='NCHW'
        ``'NCHW'`` or ``'NHWC'``.
    copy : bool, optional, default=True
        Return a new tensor or call in-place.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: %s' % data_format)
    if context.executing_eagerly():
        return OpLib.execute(
            'SpaceToDepth', inputs,
            outputs=[None] if copy else inputs,
            block_size=block_size, mode=mode.upper(), data_format=data_format)
    return OpLib.add('SpaceToDepth', inputs, block_size=block_size,
                     mode=mode.upper(), data_format=data_format, **kwargs)


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
