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

from dragon.core.tensor import Tensor
import dragon.ops as ops

__all__ = [
    'convolution',
    'relu',
    'softmax',
    'conv2d',
    'conv2d_transpose',
    'avg_pool',
    'max_pool',
    'xw_plus_b',
    'bias_add',
    'dropout',
    'sigmoid_cross_entropy_with_logits',
    'softmax_cross_entropy_with_logits',
    'sparse_softmax_cross_entropy_with_logits',
    'l2_loss'
]


def convolution(input, filter, padding, strides=None,
                dilation_rate=None, name=None, data_format=None):
    num_total_dims = filter.get_shape().ndims
    if num_total_dims is None:
        num_total_dims = input.get_shape().ndims
    if num_total_dims is None:
        raise ValueError("rank of input or filter must be known.")
    num_spatial_dims = num_total_dims - 2

    # make default parameters
    if data_format is None:
        data_format = 'NHWC'
    if strides is None:
        strides = [1] * num_total_dims
    else:
        if len(strides) != num_total_dims:
            _strides = [1] * num_total_dims
            _n_provides = len(strides)
            if data_format == 'NHWC':
                _strides[1 : 1 + _n_provides] = strides
            else:
                _strides[2 : 2 + _n_provides] = strides
            strides = _strides
    if dilation_rate is not None:
        if len(dilation_rate) != num_total_dims:
            _dilation_rate = [1] * num_total_dims
            _n_provides = len(dilation_rate)
            if data_format == 'NHWC':
                _dilation_rate[1 : 1 + _n_provides] = dilation_rate
            else:
                _dilation_rate[2 : 2 + _n_provides] = dilation_rate
            dilation_rate = _dilation_rate

    if num_spatial_dims == 2:
        return conv2d(input, filter,
                      strides, padding, dilation_rate,
                      data_format, name)
    else:
        raise NotImplementedError('conv{}d is not implemented.'.format(num_spatial_dims))


def relu(features, name=None):

   return ops.Relu(features, name=name)


def softmax(logits, dim=-1, name=None):

    return ops.Softmax(logits, axis=dim)


def conv2d(input, filter, strides, padding, dilation_rate=None,
           data_format='NHWC', name=None, **kwargs):
    """Compute 2D convolution according to the given 4D ``input`` and ``filter``.

    For **NHWC** format, filter should be as ``[filter_height, filter_width, in_channels, out_channels]``.

    For **NCHW** format, filter should be as ``[out_channels, in_channels, filter_height, filter_width]``.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    filter : Tensor
        The filter tensor.
    strides : list of int
        The strides with length 4.
    padding : str
        The padding algorithm. ``VALID`` or ``SAME``.
    dilation_rate : list of int or None
        The dilation rates with with length 4.
    data_format : str
        The data format. ``NHWC`` or ``NCHW``.
    name : str
        The optional name for this operator.

    Returns
    -------
    Tensor
        The output tensor.

    """
    if filter.shape is None:
        raise ValueError('filter must have a valid shape.')
    else:
        if len(filter.shape) != 4:
            raise ValueError('filter must be a 4D Tensor.')
    if len(strides) != 4:
        raise ValueError('strides must be a list with length 4.')
    if dilation_rate is not None:
        if len(dilation_rate) != 4:
            raise ValueError(' dilation_rate must be a list with length 4.')

    if data_format == 'NHWC':
        output = ops.Conv2d([input, filter],
                            num_output=filter.shape[3],
                            kernel_size=filter.shape[0:2],
                            stride=strides[1:3],
                            dilation=dilation_rate[1:3] if dilation_rate is not None else 1,
                            padding=padding,
                            data_format=data_format)
        return output
    elif data_format == 'NCHW':
        output = ops.Conv2d([input, filter],
                            num_output=filter.shape[0],
                            kernel_size=filter.shape[2:4],
                            stride=strides[2:4],
                            dilation=dilation_rate[2:4] if dilation_rate is not None else 1,
                            padding=padding,
                            data_format=data_format)
        return output
    else:
        raise ValueError('Unknown data format: {}'.format(data_format))


def conv2d_transpose(value, filter, output_shape, strides, padding='SAME',
                     data_format='NHWC', name=None):
    """Compute 2D deconvolution according to the given 4D ``input`` and ``filter``.

    For **NHWC** format, filter should be as ``[filter_height, filter_width, out_channels, in_channels]``.

    For **NCHW** format, filter should be as ``[in_channels, out_channels, filter_height, filter_width]``.

    ``output_shape`` will be ignored if padding algorithm is **VALID**.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    filter : Tensor
        The filter tensor.
    output_shape : list of int
        The deterministic output shape for **SAME** padding.
    strides : list of int
        The strides with length 4.
    padding : str
        The padding algorithm. ``VALID`` or ``SAME``.
    data_format : str
        The data format. ``NHWC`` or ``NCHW``.
    name : str
        The optional name for this operator.

    Returns
    -------
    Tensor
        The output tensor.

    """
    if filter.shape is None:
        raise ValueError('filter must have a valid shape.')
    else:
        if len(filter.shape) != 4:
            raise ValueError('filter must be a 4D Tensor.')
    if len(strides) != 4:
        raise ValueError('strides must be a list with length 4.')
    if not isinstance(output_shape, list):
        raise TypeError('output_shape should be a list.')
    if len(output_shape) != 4:
        raise ValueError('output_shape should be a list with length 4.')

    if data_format == 'NHWC':
        output = ops.Conv2dTranspose([value, filter],
                                     num_output=filter.shape[2],
                                     kernel_size=filter.shape[0:2],
                                     stride=strides[1:3],
                                     padding=padding,
                                     data_format=data_format,
                                     output_shape=output_shape)
        return output
    elif data_format == 'NCHW':
        output = ops.Conv2dTranspose([value, filter],
                                     num_output=filter.shape[1],
                                     kernel_size=filter.shape[2:4],
                                     stride=strides[2:4],
                                     padding=padding,
                                     data_format=data_format,
                                     output_shape=output_shape)
        return output
    else:
        raise ValueError('Unknown data format: {}'.format(data_format))


def avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None):
    """Perform avg pooling on spatial axes.

    Parameters
    ----------
    value : Tensor
        The input tensor.
    ksize : list of int
        The kernel size with length >= 4.
    strides : list of int
        The strides with length >= 4.
    padding : str
        The padding algorithm. ``VALID`` or ``SAME``.
    data_format : str
        The data format. ``NHWC`` or ``NCHW``.
    name : None or str
        The optional name of op.

    Returns
    -------
    Tensor
        The output tensor.

    """
    if len(ksize) < 4:
        raise ValueError('ksize must be a list with length >=4.')
    if len(strides) < 4:
        raise ValueError('strides must be a list with length >=4.')
    if len(ksize) != len(strides):
        raise ValueError('ksize and strides should have the same length.')
    if len(ksize) == 4:
        if data_format == 'NHWC':
            if ksize[0] != 1 or ksize[3] != 1 or strides[0] != 1 or strides[3] != 1:
                raise ValueError('The pooling can only be performed on spatial axes.')
            return ops.Pool2d(value, [ksize[1], ksize[2]], [strides[1], strides[2]],
                              padding=padding, data_format=data_format, mode='AVG')
        if data_format == 'NCHW':
            if ksize[0] != 1 or ksize[1] != 1 or strides[0] != 1 or strides[1] != 1:
                raise ValueError('The pooling can only be performed on spatial axes.')
            return ops.Pool2d(value, [ksize[2], ksize[3]], [strides[2], strides[3]],
                              padding=padding, data_format=data_format, mode='AVG')
    else:
        raise NotImplementedError('Pool{}d has not been implemented yet.'.format(len(ksize) - 2))


def max_pool(value, ksize, strides, padding, data_format='NHWC', name=None):
    """Perform max pooling on spatial axes.

    Parameters
    ----------
    value : Tensor
        The input tensor.
    ksize : list of int
        The kernel size with length >= 4.
    strides : list of int
        The strides with length >= 4.
    padding : str
        The padding algorithm. ``VALID`` or ``SAME``.
    data_format : str
        The data format. ``NHWC`` or ``NCHW``.
    name : None or str
        The optional name of op.

    Returns
    -------
    Tensor
        The output tensor.

    """
    if len(ksize) < 4:
        raise ValueError('ksize must be a list with length >=4.')
    if len(strides) < 4:
        raise ValueError('strides must be a list with length >=4.')
    if len(ksize) != len(strides):
        raise ValueError('ksize and strides should have the same length.')
    if len(ksize) == 4:
        if data_format == 'NHWC':
            if ksize[0] != 1 or ksize[3] != 1 or strides[0] != 1 or strides[3] != 1:
                raise ValueError('The pooling can only be performed on spatial axes.')
            return ops.Pool2d(value, [ksize[1], ksize[2]], [strides[1], strides[2]],
                              padding=padding, data_format=data_format, mode='MAX')
        if data_format == 'NCHW':
            if ksize[0] != 1 or ksize[1] != 1 or strides[0] != 1 or strides[1] != 1:
                raise ValueError('The pooling can only be performed on spatial axes.')
            return ops.Pool2d(value, [ksize[2], ksize[3]], [strides[2], strides[3]],
                              padding=padding, data_format=data_format, mode='MAX')
    else:
        raise NotImplementedError('Pool{}d has not been implemented yet.'.format(len(ksize) - 2))


def xw_plus_b(x, weights, biases, name=None):

    if weights.shape is None:
        raise ValueError('weights must have a valid shape.')
    else:
        if len(weights.shape) != 2:
            raise ValueError('weights must be a 2D Tensor')

    if biases.shape is None:
        raise ValueError('biases must a have a valid shape.')
    else:
        if len(biases.shape) != 1:
            raise ValueError('biases must be a 1D Tensor')
        if weights.shape[1] != biases.shape[0]:
            raise ValueError('the shape of weights and biaes are incompatible.')

    return ops.InnerProduct([x, weights, biases], num_output=weights.shape[1], TransW=False)


def bias_add(value, bias, data_format='NHWC', name=None):
    return ops.BiasAdd([value, bias], data_format=data_format)

def sigmoid_cross_entropy_with_logits(logits, targets, name=None):

    return ops.SigmoidCrossEntropy([logits, targets], normalization='UNIT', name=None)


def softmax_cross_entropy_with_logits(_sentinel=None,
                                      labels=None, logits=None,
                                      dim=-1, name=None):

    if _sentinel is not None:
        raise ValueError('Only call `softmax_cross_entropy_with_logits` '
                         'with named arguments (labels=..., logits=..., ...)')

    if dim == -1: dim = 1
    return ops.SoftmaxCrossEntropy([logits, labels], axis=dim, normalization='UNIT', name=name)


def sparse_softmax_cross_entropy_with_logits(logits, labels, dim=-1, name=None):

    if dim == -1: dim = 1
    return ops.SparseSoftmaxCrossEntropy([logits, labels], axis=dim, normalization='UNIT', name=name)


def l2_loss(t, name=None):
    return (ops.Reduce(ops.Square(t), operation='SUM') * 0.5)


def dropout(x, keep_prob, name=None):
    return ops.Dropout(x, 1 - keep_prob)
