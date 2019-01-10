# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/normalization.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon

from dragon.vm.tensorflow.framework import tensor_shape
from dragon.vm.tensorflow.layers import base
from dragon.vm.tensorflow.ops import init_ops


class BatchNormalization(base.Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer=init_ops.zeros_initializer(),
                 gamma_initializer=init_ops.ones_initializer(),
                 moving_mean_initializer=init_ops.zeros_initializer(),
                 moving_variance_initializer=init_ops.ones_initializer(),
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 fused=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(BatchNormalization, self).__init__(trainable=trainable, name=name, **kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.renorm = renorm
        self.fused = fused
        self.trainable = trainable
        if renorm:
            raise ValueError('renorm is currently not supported.')

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndim = input_shape.ndims
        if self.axis < 0:
            axis = ndim + self.axis
        else:
            axis = self.axis
        if axis < 0 or axis >= ndim:
            raise ValueError('Value of `axis` argument ' + str(self.axis) +
                             ' is out of range for input with rank ' + str(ndim))
        if axis + 1 == ndim:
            self._data_format = 'NHWC'
        elif axis == 1:
            self._data_format = 'NCHW'
        else:
            raise ValueError(
                'Only axis 1 or last axis are currently supported dimensions for '
                    'batch norm. Got `axis` dimension: ', axis)

        param_dim = input_shape[axis]
        if not param_dim.value:
            raise ValueError('Input has undefined `axis` dimension. Input shape: ', input_shape)
        self.input_spec = base.InputSpec(ndim=ndim, axes={self.axis: param_dim.value})

        self.moving_mean = self.add_variable(
            name='moving_mean',
            shape=(param_dim.value,),
            initializer=self.moving_mean_initializer,
            trainable=False)

        self.moving_variance = self.add_variable(
            name='moving_variance',
            shape=(param_dim.value,),
            initializer=self.moving_variance_initializer,
            trainable=False)

        self.gamma = self.add_variable(
            name='gamma',
            shape=(param_dim.value,),
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            trainable=self.scale)

        self.beta = self.add_variable(
            name='beta',
            shape=(param_dim.value,),
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            trainable=self.center)

        self.built = True

    def call(self, inputs, training=False, *args, **kwargs):
        use_stats = 0 if training else 1
        return dragon.ops.BatchNorm([
            inputs,
            self.moving_mean,
            self.moving_variance,
            self.gamma,
            self.beta],
            axis=self.axis,
            momentum=self.momentum,
            eps=self.epsilon,
            use_stats=use_stats)


def batch_normalization(
    inputs,
    axis=-1,
    momentum=0.99,
    epsilon=1e-3,
    center=True,
    scale=True,
    beta_initializer=init_ops.zeros_initializer(),
    gamma_initializer=init_ops.ones_initializer(),
    moving_mean_initializer=init_ops.zeros_initializer(),
    moving_variance_initializer=init_ops.ones_initializer(),
    beta_regularizer=None,
    gamma_regularizer=None,
    training=False,
    trainable=True,
    name=None,
    reuse=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=False):
    return BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=fused,
        trainable=trainable,
        name=name,
        _reuse=reuse,
        _scope=name).apply(inputs, training=training)


# Aliases
BatchNorm = BatchNormalization
batch_norm = batch_normalization