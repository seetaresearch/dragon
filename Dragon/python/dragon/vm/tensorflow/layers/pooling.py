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
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py>
#
# ------------------------------------------------------------

from dragon.vm.tensorflow.ops import nn
from dragon.vm.tensorflow.layers import base, utils


class _Pooling2D(base.Layer):
    def __init__(self, pool_function, pool_size, strides,
                 padding='valid', data_format='channels_last',
                 name=None, **kwargs):
        super(_Pooling2D, self).__init__(name=name, **kwargs)
        self.pool_function = pool_function
        self.pool_size = utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = utils.normalize_tuple(strides, 2, 'strides')
        self.padding = utils.normalize_padding(padding)
        self.data_format = utils.normalize_data_format(data_format)
        self.input_spec = base.InputSpec(ndim=4)

    def call(self, inputs):
        if self.data_format == 'channels_last':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        outputs = self.pool_function(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format, 4))
        return outputs


class AveragePooling2D(_Pooling2D):
    def __init__(self, pool_size, strides,
                 padding='valid', data_format='channels_last',
                 name=None, **kwargs):
        super(AveragePooling2D, self).__init__(
            nn.avg_pool,
            pool_size=pool_size, strides=strides, padding=padding,
            data_format=data_format, name=name, **kwargs)


def average_pooling2d(inputs,
                      pool_size, strides,
                      padding='valid', data_format='channels_last',
                      name=None):
    layer = AveragePooling2D(pool_size=pool_size, strides=strides,
                             padding=padding, data_format=data_format,
                             name=name)
    return layer.apply(inputs)


class MaxPooling2D(_Pooling2D):
    def __init__(self, pool_size, strides,
                 padding='valid', data_format='channels_last',
                 name=None, **kwargs):
        super(MaxPooling2D, self).__init__(
            nn.max_pool,
            pool_size=pool_size, strides=strides, padding=padding,
            data_format=data_format, name=name, **kwargs)


def max_pooling2d(inputs,
                      pool_size, strides,
                      padding='valid', data_format='channels_last',
                      name=None):
    layer = MaxPooling2D(pool_size=pool_size, strides=strides,
                         padding=padding, data_format=data_format,
                         name=name)
    return layer.apply(inputs)