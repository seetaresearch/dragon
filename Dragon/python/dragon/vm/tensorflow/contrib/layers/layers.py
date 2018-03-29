# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes based on:
#
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layer.py>
#
# ------------------------------------------------------------

from collections import defaultdict

import dragon.ops as op_lib

import dragon.vm.tensorflow.framework.ops as ops
from dragon.vm.tensorflow.contrib.layers import initializers
from dragon.vm.tensorflow.ops import init_ops
from dragon.vm.tensorflow.ops import nn
from dragon.vm.tensorflow.ops import var_scope as vs
from dragon.vm.tensorflow.layers import layers
from dragon.vm.tensorflow.layers import normalization as normalization_layers

__all__ = ['flatten']

_LAYERS_UID_DICT = defaultdict(int)

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


def _default_scope(scope, key, indicator):
    if scope is None:
        return indicator
        # global _LAYERS_UID_DICT
        # _LAYERS_UID_DICT[key] += 1
        # return '{}{}'.format(indicator, _LAYERS_UID_DICT[key])
    else:
        return scope


def avg_pool2d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               data_format=DATA_FORMAT_NHWC,
               outputs_collections=None,
               scope=None):
    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
        raise ValueError('data_format has to be either NCHW or NHWC.')
    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    return layers.average_pooling2d(inputs=inputs,
                                    pool_size=kernel_size,
                                    strides=stride,
                                    padding=padding,
                                    data_format=df)


def max_pool2d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               data_format=DATA_FORMAT_NHWC,
               outputs_collections=None,
               scope=None):
    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
        raise ValueError('data_format has to be either NCHW or NHWC.')
    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    return layers.max_pooling2d(inputs=inputs,
                                pool_size=kernel_size,
                                strides=stride,
                                padding=padding,
                                data_format=df)


def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
    scope = _default_scope(scope, 'CONVOLUTION', 'Conv')
    if data_format not in [None, 'NHWC', 'NCHW']:
        raise ValueError('Invalid data_format: %r' % (data_format,))
    data_format = 'channels_first' if data_format == 'NCHW' else 'channels_last'
    input_rank = inputs.get_shape().ndims

    with vs.variable_scope(scope, reuse=reuse) as sc:
        if input_rank == 4:
            return layers.conv2d(
                inputs=inputs,
                filters=num_outputs,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                data_format=data_format,
                dilation_rate=rate,
                activation=activation_fn,
                use_bias=True if biases_initializer is not None else False,
                kernel_initializer=weights_initializer,
                bias_initializer=biases_initializer,
                bias_regularizer=biases_regularizer,
                activity_regularizer=None,
                trainable=trainable,
                reuse=reuse)


# Simple alias.
convolution2d = convolution
conv2d = convolution2d


def fully_connected(inputs,
                    num_outputs,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):
    scope = _default_scope(scope, 'FULLY_CONNECTED', 'fully_connected')
    with vs.variable_scope(scope, reuse=reuse) as sc:
        return layers.dense(
            inputs=inputs,
            units=num_outputs,
            activation=activation_fn,
            use_bias=True if biases_initializer is not None else False,
            kernel_initializer=weights_initializer,
            bias_initializer=biases_initializer,
            bias_regularizer=biases_regularizer,
            activity_regularizer=None,
            trainable=trainable,
            reuse=reuse)


def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               param_initializers=None,
               param_regularizers=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               batch_weights=None,
               fused=False,
               data_format=DATA_FORMAT_NHWC,
               zero_debias_moving_mean=False,
               scope=None,
               renorm=False,
               renorm_clipping=None,
               renorm_decay=0.99):
    scope = _default_scope(scope, 'BATCH_NORM', 'BatchNorm')
    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
        raise ValueError('data_format has to be either NCHW or NHWC.')
    axis = 1 if data_format == DATA_FORMAT_NCHW else -1

    with vs.variable_scope(scope, reuse=reuse) as sc:
        if not param_initializers:
            param_initializers = {}
        beta_initializer = param_initializers.get('beta', init_ops.zeros_initializer())
        gamma_initializer = param_initializers.get('gamma', init_ops.ones_initializer())
        moving_mean_initializer = param_initializers.get('moving_mean', init_ops.zeros_initializer())
        moving_variance_initializer = param_initializers.get('moving_variance', init_ops.ones_initializer())

        if not param_regularizers:
            param_regularizers = {}

        beta_regularizer = param_regularizers.get('beta')
        gamma_regularizer = param_regularizers.get('gamma')

        return layers.batch_norm(
            inputs=inputs,
            axis=axis,
            momentum=decay,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            trainable=trainable,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_decay,
            fused=fused,
            training=is_training)


def flatten(inputs,
            outputs_collections=None,
            scope=None):
    return op_lib.Flatten(inputs, axis=0, keep_axes=2)

