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
from dragon.core.ops import array_ops
from dragon.core.ops import loss_ops
from dragon.core.ops import normalization_ops
from dragon.core.ops import vision_ops
from dragon.core.util import nest
from dragon.core.util import six


def avg_pool(
    input,
    ksize,
    strides,
    padding,
    data_format=None,
    name=None,
):
    """Apply the n-dimension average pooling.

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Sequence[int]
        The size(s) of sliding window.
    strides : Sequence[int], optional
        The stride(s) of sliding window.
    padding : Union[{'VALID', 'SAME'}, Sequence[int]]
        The padding algorithm or padding sizes.
    data_format : {'NCHW', 'NCDHW', 'NHWC', 'NDHWC'}, optional
        The optional data format.
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
    normalize_spatial_args = \
        functools.partial(
            _normalize_spatial_args,
            num_total_dims=num_total_dims,
            num_spatial_dims=num_spatial_dims,
            start_axis=start_axis,
        )
    ksize = normalize_spatial_args('ksize', ksize)
    strides = normalize_spatial_args('strides', strides)
    padding, pads = normalize_spatial_args('padding', padding)
    return getattr(vision_ops, 'pool{}d'.format(num_spatial_dims))(
        [input],
        kernel_shape=ksize.shape[start_axis:start_axis + num_spatial_dims],
        strides=strides[start_axis:start_axis + num_spatial_dims],
        padding=padding,
        pads=pads,
        data_format='NCHW' if data_format.startswith('NC') else 'NHWC',
        name=name,
        mode='AVG',
    )


def avg_pool2d(
    input,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None,
):
    """Apply the 2d average pooling.

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Sequence[int]
        The size(s) of sliding window.
    strides : Sequence[int], optional
        The stride(s) of sliding window.
    padding : Union[{'VALID', 'SAME'}, Sequence[int]]
        The padding algorithm or padding sizes.
    data_format : {'NCHW', 'NCDHW', 'NHWC', 'NDHWC'}, optional
        The optional data format.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    start_axis = 2 if data_format.startswith('NC') else 1
    ksize = _normalize_spatial_args('ksize', ksize, 4, 2, start_axis)
    return avg_pool(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )


def bias_add(value, bias, data_format='NHWC', name=None):
    return vision_ops.bias_add(
        [value, bias],
        data_format=data_format,
        name=name,
    )


def convolution(
    input,
    filters,
    strides=None,
    padding='VALID',
    data_format=None,
    dilations=None,
    name=None,
    **kwargs
):
    """Apply the n-dimension convolution.

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The weight tensor.
    strides : Sequence[int], optional
        The stride(s) of sliding window.
    padding : Union[{'VALID', 'SAME'}, Sequence[int]], optional
        The padding algorithm or padding size(s).
    data_format : {'NCHW', 'NCDHW', 'NHWC', 'NDHWC'}, optional
        The optional data format.
    dilations : Sequence[int], optional
        The rate(s) of dilated kernel.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    if filters.shape is not None:
        num_total_dims = len(filters.shape)
    else:
        raise ValueError('Rank of <filters> must be determined.')
    num_spatial_dims = num_total_dims - 2
    data_format = data_format if data_format else 'NHWC'
    start_axis = 2 if data_format.startswith('NC') else 1
    normalize_spatial_args = \
        functools.partial(
            _normalize_spatial_args,
            num_total_dims=num_total_dims,
            num_spatial_dims=num_spatial_dims,
            start_axis=start_axis,
        )
    strides = normalize_spatial_args('strides', strides)
    dilations = normalize_spatial_args('dilations', dilations)
    padding, pads = normalize_spatial_args('padding', padding)
    return getattr(vision_ops, '{}{}d'.format(
        kwargs.get('conv_type', 'conv'), num_spatial_dims))(
            [input, filters],
            kernel_shape=filters.shape[2:],
            strides=strides[start_axis:start_axis + num_spatial_dims],
            dilations=dilations[start_axis:start_axis + num_spatial_dims],
            padding=padding,
            pads=pads,
            data_format='NCHW' if data_format.startswith('NC') else 'NHWC',
            name=name)


def conv_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding='SAME',
    data_format=None,
    dilations=None,
    name=None,
):
    """Apply the n-dimension deconvolution.

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The weight tensor.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The determined shape of output.
    strides : Sequence[int]
        The stride(s) of sliding window.
    padding : Union[{'VALID', 'SAME'}, Sequence[int]], optional
        The padding algorithm or padding size(s).
    data_format : {'NCHW', 'NCDHW', 'NHWC', 'NDHWC'}, optional
        The optional data format.
    dilations : Sequence[int], optional
        The rate(s) of dilated kernel.
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
    normalize_spatial_args = \
        functools.partial(
            _normalize_spatial_args,
            num_total_dims=num_total_dims,
            num_spatial_dims=num_spatial_dims,
            start_axis=start_axis,
        )
    strides = normalize_spatial_args('strides', strides)
    dilations = normalize_spatial_args('dilations', dilations)
    padding, pads = normalize_spatial_args('padding', padding)
    if padding == 'SAME' and output_shape is None:
        raise ValueError('Excepted <output_shape> for same padding.')
    output_shape = normalize_spatial_args('output_shape', output_shape)
    return getattr(vision_ops, 'conv{}d_transpose'.format(num_spatial_dims))(
        [input, filters],
        kernel_shape=filters.shape[2:],
        strides=strides[start_axis:start_axis + num_spatial_dims],
        dilations=dilations[start_axis:start_axis + num_spatial_dims],
        padding=padding,
        output_shape=output_shape,
        pads=pads,
        data_format='NCHW' if data_format.startswith('NC') else 'NHWC',
        name=name,
    )


def conv2d(
    input,
    filters,
    strides,
    padding,
    data_format='NHWC',
    dilations=None,
    name=None,
):
    """Apply the 2d convolution.

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The weight tensor.
    strides : Sequence[int]
        The stride(s) of sliding window.
    padding : Union[{'VALID', 'SAME'}, Sequence[int]]
        The padding algorithm or padding sizes.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.
    dilations : Sequence[int], optional
        The rate(s) of dilated kernel.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return convolution(**locals())


def conv2d_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding='SAME',
    data_format='NHWC',
    dilations=None,
    name=None,
):
    """Apply the 2d deconvolution.

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The weight tensor.
    output_shape : Union[Sequence[int], dragon.Tensor], optional
        The determined shape of output.
    strides : Sequence[int]
        The stride(s) of sliding window.
    padding : Union[{'VALID', 'SAME'}, Sequence[int]]
        The padding algorithm or padding sizes.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.
    dilations : Sequence[int], optional
        The rate(s) of dilated kernel.
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
    strides,
    padding,
    data_format='NHWC',
    dilations=None,
    name=None,
):
    """Apply the 2d depthwise convolution.
    `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.

    Parameters
    ----------
    input : dragon.Tensor
       The input tensor.
    filters : dragon.Tensor
       The weight tensor.
    strides : Sequence[int]
        The stride(s) of sliding window.
    padding : Union[{'VALID', 'SAME'}, Sequence[int]]
        The padding algorithm or padding sizes.
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.
    dilations : Sequence[int], optional
        The rate(s) of dilated kernel.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return convolution(conv_type='depthwise_conv', **locals())


def dropout(x, rate, name=None, **kwargs):
    r"""Set the elements of the input to zero randomly.
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
        The dropping ratio.
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


def l2_loss(t, name=None):
    return loss_ops.l2_loss(t, normalization='NONE', name=name)


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
    data_format : {'NCHW', 'NHWC'}, optional
        The optional data format.
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
    padding,
    data_format=None,
    name=None,
):
    """Apply the n-dimension max pooling.

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Sequence[int]
        The size(s) of sliding window.
    strides : Sequence[int]
        The stride(s) of sliding window.
    padding : Union[{'valid', 'same'}, Sequence[int]]
        The padding algorithm or padding sizes.
    data_format : {'NCHW', 'NCDHW', 'NHWC', 'NDHWC'}, optional
        The optional data format.
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
    normalize_spatial_args = \
        functools.partial(
            _normalize_spatial_args,
            num_total_dims=num_total_dims,
            num_spatial_dims=num_spatial_dims,
            start_axis=start_axis,
        )
    ksize = normalize_spatial_args('ksize', ksize)
    strides = normalize_spatial_args('strides', strides)
    padding, pads = normalize_spatial_args('padding', padding)
    return getattr(vision_ops, 'pool{}d'.format(num_spatial_dims))(
        [input],
        kernel_shape=ksize[start_axis:start_axis + num_spatial_dims],
        strides=strides[start_axis:start_axis + num_spatial_dims],
        padding=padding,
        pads=pads,
        data_format='NCHW' if data_format.startswith('NC') else 'NHWC',
        name=name,
        mode='MAX',
    )


def max_pool2d(
    input,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None,
):
    """Apply the 2d max pooling.

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    ksize : Sequence[int]
        The size(s) of sliding window.
    strides : Sequence[int], optional
        The stride(s) of sliding window.
    padding : Union[{'valid', 'same'}, Sequence[int]]
        The padding algorithm or padding size(s).
    data_format : {'NCHW', 'NCDHW', 'NHWC', 'NDHWC'}, optional
        The optional data format.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    start_axis = 2 if data_format.startswith('NC') else 1
    ksize = _normalize_spatial_args('ksize', ksize, 4, 2, start_axis)
    return max_pool(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )


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


def sigmoid_cross_entropy_with_logits(logits, targets, name=None):
    return loss_ops.sigmoid_cross_entropy(
        [logits, targets],
        normalization='UNIT',
        name=name,
    )


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
    """Compute the softmax cross entropy with contiguous labels.

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
    return loss_ops.softmax_cross_entropy(
        [logits, labels],
        axis=-1,
        reduction='none',
        name=name,
    )


def sparse_softmax_cross_entropy_with_logits(labels, logits, name=None):
    """Compute the softmax cross entropy with sparse labels.

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
    return loss_ops.sparse_softmax_cross_entropy(
        [logits, labels],
        axis=-1,
        reduction='none',
        name=name,
    )


def top_k(input, k=1, sorted=True, name=None):
    """Return the top-K largest elements along the last axis.

    Parameters
    ----------
    input : dragon.Tensor
        The input tensor.
    k : int, optional, default=1
        The number of top elements to select.
    sorted : bool, optional
        Whether to return in the sorted order.
    name : str, optional
        The operation name.

    Returns
    -------
    Sequence[dragon.vm.torch.Tensor]
        The value and index tensor.

    """
    return array_ops.top_k(input, k=k, sorted=sorted, name=name)


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
                defaults[start_axis:start_axis + n_provides] = values
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
                for i in range(start_axis, start_axis + num_spatial_dims):
                    pads_l.append(padding_tuple[i * 2])
                    pads_r.append(padding_tuple[i * 2 + 1])
                pads = pads_l + pads_r
            else:
                raise ValueError(
                    'Except 1, {} or {} values if <padding> set as explict pads.'
                    .format(num_spatial_dims, num_spatial_dims * 2)
                )
        return padding, pads
    elif name == 'output_shape':
        if values is not None:
            if types.is_tensor(values):
                values_wide, is_eager = [], types.is_eager_tensor(values)
                for i in range(start_axis, start_axis + num_spatial_dims):
                    values_wide.append(int(values[i]) if is_eager else values[i])
                return values_wide
            else:
                values = nest.flatten(values)
                if len(values) == num_spatial_dims:
                    return values
                elif len(values) == num_total_dims:
                    return values[start_axis:start_axis + start_axis + num_spatial_dims]
                else:
                    raise ValueError(
                        'Except {} or {} values for <output_shape>.'
                        .format(num_spatial_dims, num_spatial_dims * 2)
                    )
        return values
