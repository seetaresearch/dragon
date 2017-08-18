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

   return ops.Relu(features, name=name)


def softmax(logits, dim=-1, name=None):

    return ops.Softmax(logits, axis=dim)


def conv2d(input, filter, strides, pads=(0, 0, 0, 0),
           use_cudnn_on_gpu=True, padding=None,
           data_format='NCHW', name=None):

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

    return ops.BiasAdd([value, bias], data_format=data_format, name=None)

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