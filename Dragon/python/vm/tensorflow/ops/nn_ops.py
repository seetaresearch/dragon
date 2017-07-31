# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = [
    'relu',
    'softmax',
    'conv2d',
    'avg_pool',
    'max_pool',
    'xw_plus_b',
    'bias_add',
    'sigmoid_cross_entropy_with_logits',
    'softmax_cross_entropy_with_logits',
    'sparse_softmax_cross_entropy_with_logits',
    'l2_loss'
]

import dragon.ops as ops


def relu(features, name=None):
   """
   Computes Rectified Linear: `max(features, 0)`.

    Args:
      features: A `Tensor`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` with the same type.
   """

   return ops.Relu(features, name=name)


def softmax(logits, dim=-1, name=None):
    """
    Computes softmax activations.

      For each batch `i` and class `j` we have

      softmax = exp(logits) / reduce_sum(exp(logits), dim)

     Args:
       logits: A non-empty `Tensor`.
       dim: The dimension softmax would be performed on. The default is -1 which
            indicates the last dimension.
       name: A name for the operation (optional).

     Returns:
        A `Tensor`. Has the same type as `logits`. Same shape as `logits`.

    """

    return ops.Softmax(logits, axis=dim)


def conv2d(input, filter, strides, pads=(0, 0, 0, 0),
           use_cudnn_on_gpu=True, padding=None,
           data_format='NCHW', name=None):
    """
    Computes a 2-D convolution given 4-D input and filter tensors.

     Args:
       input: A Tensor.
       filter: A Tensor.
               For 'NCHW', shape as [out_channels, in_channels, filter_height, filter_width].
               For 'NHWC', shape as [filter_height, filter_width, in_channels, out_channels].
       strides: A list of ints. 1-D of length 4.
                The stride of the sliding window for each dimension of input.
       pads: A list of ints. 1-D of length 4.
       use_cudnn_on_gpu: An optional bool. Defaults to True.
       padding: A string from: "SAME", "VALID". (deprecated)
       data_format: A string. 'NHWC' and 'NCHW' are supported.
       name: A name for the operation (optional).

     Returns:
        A Tensor. Has the same type as input.
    """

    if filter.shape is None:
        raise ValueError('filter must have a valid shape.')
    else:
        if len(filter.shape) != 4:
            raise ValueError('filter must be a 4D Tensor')
    if len(strides) != 4:
        raise ValueError('strides must be a list of length 4.')

    if data_format == 'NCHW':
        output = ops.Conv2D([input, filter],
                            num_output=filter.shape[0],
                            kernel_size=filter.shape[2:],
                            stride=strides[2:],
                            pad=pads[2:])
        return output

    else: raise NotImplementedError()


def avg_pool(value, ksize, strides, pads=(0, 0, 0, 0),
             padding=None, data_format="NCHW", name=None):
    """
    Performs the avg pooling on the input.

      Args:
        value: A 4-D `Tensor` with type `tf.float32`.
        ksize: A list of ints that has length 4.
        strides: A list of ints that has length 4.
        pads: A list of ints or a int.
        padding: A string, either `'VALID'` or `'SAME'`. (deprecated)
        data_format: A string. 'NHWC' and 'NCHW' are supported.
        name: Optional name for the operation.

      Returns:
        A `Tensor` with type `tf.float32`. The avg pooled output tensor.
    """

    if len(strides) != 4:
        raise ValueError('strides must be a list of length 4.')

    if len(ksize) != 4:
        raise ValueError('strides must be a list of length 4.')

    if data_format == 'NCHW':
        if pads is None: pads = 0
        return ops.Pool2D(value,
                          kernel_size=ksize[2:],
                          stride=strides[2:],
                          pad=pads,
                          mode='AVG_POOLING')

    else: raise NotImplementedError()


def max_pool(value, ksize, strides, pads=(0, 0, 0, 0),
             padding=None, data_format="NCHW", name=None):
    """
    Performs the max pooling on the input.

      Args:
        value: A 4-D `Tensor` with type `tf.float32`.
        ksize: A list of ints that has length 4.
        strides: A list of ints that has length 4.
        pads: A list of ints or a int.
        padding: A string, either `'VALID'` or `'SAME'`. (deprecated)
        data_format: A string. 'NHWC' and 'NCHW' are supported.
        name: Optional name for the operation.

      Returns:
        A `Tensor` with type `tf.float32`. The max pooled output tensor.
    """

    if len(strides) != 4:
        raise ValueError('strides must be a list of length 4.')

    if len(ksize) != 4:
        raise ValueError('strides must be a list of length 4.')

    if data_format == 'NCHW':
        if pads is None: pads = 0
        return ops.Pool2D(value,
                          kernel_size=ksize[2:],
                          stride=strides[2:],
                          pad=pads,
                          mode='MAX_POOLING')

    else: raise NotImplementedError()


def xw_plus_b(x, weights, biases, name=None):
    """
    Computes matmul(x, weights) + biases.

      Args:
        x: a 2D tensor.  Dimensions typically: batch, in_units
        weights: a 2D tensor.  Dimensions typically: in_units, out_units
        biases: a 1D tensor.  Dimensions: out_units
        name: A name for the operation (optional).  If not specified
          "xw_plus_b" is used.

      Returns:
        A 2-D Tensor computing matmul(x, weights) + biases.
        Dimensions typically: batch, out_units.
    """

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


def bias_add(value, bias, data_format='NCHW', name=None):
    """
    Adds `bias` to `value`.

      This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
      Broadcasting is supported, so `value` may have any number of dimensions.
      Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
      case where both types are quantized.

      Args:
        value: A `Tensor`.
        bias: A 1-D `Tensor` with size matching the last dimension of `value`.
        data_format: A string. 'NHWC' and 'NCHW' are supported.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` with the same type as `value`.
    """

    return ops.BiasAdd([value, bias], data_format=data_format, name=None)

def sigmoid_cross_entropy_with_logits(logits, targets, name=None):
    """
    Computes sigmoid cross entropy given logits.

      Measures the probability error in discrete classification tasks in which
      each class is independent and not mutually exclusive.
      For instance, one could perform multilabel classification where a picture
      can contain both an elephant and a dog at the same time.

      Args:
        logits: A Tensor of type float32 or float64.
        targets: A Tensor of the same type and shape as logits.
        name: A name for the operation (optional).

      Returns:
        A Tensor. Has the same type as logits. Same shape as logits.
    """

    return ops.SigmoidCrossEntropyLoss([logits, targets], normalization='UNIT', name=None)


def softmax_cross_entropy_with_logits(_sentinel=None,
                                      labels=None, logits=None,
                                      dim=-1, name=None):
    """
    Computes softmax cross entropy between `logits` and `labels`.

      Measures the probability error in discrete classification tasks in which the
      classes are mutually exclusive (each entry is in exactly one class).  For
      example, each CIFAR-10 image is labeled with one and only one label: an image
      can be a dog or a truck, but not both.

      **NOTE:**  While the classes are mutually exclusive, their probabilities
      need not be.  All that is required is that each row of `labels` is
      a valid probability distribution.  If they are not, the computation of the
      gradient will be incorrect.

      If using exclusive `labels` (wherein one and only
      one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

      **WARNING:** This op expects unscaled logits, since it performs a `softmax`
      on `logits` internally for efficiency.  Do not call this op with the
      output of `softmax`, as it will produce incorrect results.

      `logits` and `labels` must have the same shape `[batch_size, num_classes]`
      and the same dtype (either `float16`, `float32`, or `float64`).

      **Note that to avoid confusion, it is required to pass only named arguments to
      this function.**

      Args:
        _sentinel: Used to prevent positional parameters. Internal, do not use.
        labels: Each row `labels[i]` must be a valid probability distribution.
        logits: Unscaled log probabilities.
        dim: The class dimension. Defaulted to -1 which is the second dimension.
        name: A name for the operation (optional).

      Returns:
        A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
        softmax cross entropy loss.
      """

    if _sentinel is not None:
        raise ValueError('Only call `softmax_cross_entropy_with_logits` '
                         'with named arguments (labels=..., logits=..., ...)')

    if dim == -1: dim = 1
    return ops.SoftmaxCrossEntropyLoss([logits, labels], axis=dim, normalization='UNIT', name=name)


def sparse_softmax_cross_entropy_with_logits(logits, labels, dim=-1, name=None):
    """
    Computes sparse softmax cross entropy between `logits` and `labels`.

      Args:
        logits: Unscaled log probabilities.
        labels: A `Tensor` of shape [batchsize,].
                Note that it is not a one-hot represention.
        dim: The class dimension. Defaulted to -1 which is the second dimension.
        name: A name for the operation (optional).

      Returns:
        A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
        sparse softmax cross entropy loss.
      """

    if dim == -1: dim = 1
    return ops.SoftmaxLoss([logits, labels], axis=dim, normalization='UNIT', name=name)


def l2_loss(t, name=None):
    """
    Computes half the L2 norm of a tensor without the sqrt:

      output = sum(t ** 2) / 2

      Args:
        t:  A Tensor. Typically 2-D, but may have any dimensions.
        name: Optional name for the operation.

      Returns:
        A Tensor. Has the same type as t. 0-D.

    """

    return (ops.Reduce(ops.Square(t), operation='SUM') * 0.5)


def dropout(x, keep_prob, name=None):
    """
    Computes dropout.

      With probability `keep_prob`, outputs the input element scaled up by
      `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
      sum is unchanged.

      Args:
        x: A tensor.
        keep_prob: A float. The probability that each element is kept.
        name: A name for this operation (optional).

      Returns:
        A Tensor of the same shape of `x`.
    """

    return ops.Dropout(x, 1 - keep_prob)


def batch_normalization(x, mean, variance,
                        offset, scale,
                        decay=0.9,
                        variance_epsilon=1e-3,
                        use_global_stats=-1,
                        name=None):
    norm_x = ops.BatchNorm([x, mean, variance], decay, variance_epsilon, use_global_stats, name=name)
    return ops.Scale([norm_x, scale, offset], name=name + '_scale' if name is not None else name)


def batch_norm_with_global_normalization(t, m, v,
                                         beta, gamma,
                                         decay=0.9,
                                         variance_epsilon=1e-3,
                                         scale_after_normalization=True,
                                         use_global_stats=-1,
                                         name=None):
    norm_x = ops.BatchNorm([t, m, v], decay, variance_epsilon, use_global_stats, name=name)
    if scale_after_normalization:
        return ops.Scale([norm_x, gamma, beta], name=name + '_scale' if name is not None else name)
    else: return norm_x