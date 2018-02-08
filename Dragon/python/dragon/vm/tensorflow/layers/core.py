# --------------------------------------------------------
# TensorFlow @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.vm.tensorflow.framework import tensor_shape
from dragon.vm.tensorflow.layers import base, utils
from dragon.vm.tensorflow.ops import init_ops
from dragon.vm.tensorflow.ops import nn, standard_ops


class Dense(base.Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Dense, self).__init__(trainable=trainable, name=name, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        dtype=self.dtype,
                                        trainable=True)
        if self.use_bias:
            self.bias = self.add_variable('bias',
                                          shape=[self.units, ],
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          dtype=self.dtype,
                                          trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        shape = inputs.get_shape().as_list()
        output_shape = shape[:-1] + [self.units]
        if len(output_shape) > 2:
            raise NotImplementedError()
        else:
            outputs = standard_ops.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def dense(inputs, units,
          activation=None,
          use_bias=True,
          kernel_initializer=None,
          bias_initializer=init_ops.zeros_initializer(),
          kernel_regularizer=None,
          bias_regularizer=None,
          activity_regularizer=None,
          trainable=True,
          name=None,
          reuse=None):
    layer = Dense(units,
                  activation=activation,
                  use_bias=use_bias,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer,
                  activity_regularizer=activity_regularizer,
                  trainable=trainable,
                  name=name,
                  _scope=name,
                  _reuse=reuse)
    return layer.apply(inputs)
