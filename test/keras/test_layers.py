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
"""Test layers module."""

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import tensorflow as tf


class TestLayers(unittest.TestCase):
    """Test layers."""

    def test_activation_layers(self):
        x = tf.zeros((2, 3, 3))
        _ = tf.keras.layers.ELU(alpha=1.0)(x)
        _ = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        _ = tf.keras.layers.ReLU(max_value=6.0)(x)
        _ = tf.keras.layers.ReLU(negative_slope=0.2)(x)
        _ = tf.keras.layers.SELU()(x)
        _ = tf.keras.layers.Softmax(axis=-1)(x)
        for max_value, slope in ((1., None), (6.0, 0.2)):
            try:
                _ = tf.keras.layers.ReLU(
                    max_value=max_value, negative_slope=slope)(x)
            except ValueError:
                pass

    def test_core_layers(self):
        x = tf.zeros((2, 3, 3))
        _ = tf.keras.layers.Activation('relu')(x)
        _ = tf.keras.layers.Activation('relu', inplace=True)(x)
        _ = tf.keras.layers.Dense(4)(x)
        _ = tf.keras.layers.Dense(4, use_bias=False)(x)
        _ = tf.keras.layers.Dropout(0.5)(x, training=True)
        _ = tf.keras.layers.Dropout(0.5)(x, training=False)
        try:
            _ = tf.keras.layers.Dense(4, dtype=tf.int8)(x)
        except TypeError:
            pass

    def test_conv_layers(self):
        x1 = tf.zeros((2, 3, 4))
        x2 = tf.zeros((2, 3, 4, 5))
        x3 = tf.zeros((2, 3, 4, 5, 6))
        _ = tf.keras.layers.Conv2D(3, 3, 1, padding=1, use_bias=False)(x2)
        _ = tf.keras.layers.Conv2DTranspose(3, 3, 1, padding=1, use_bias=False)(x2)
        for conv_type in ('Conv{}D', 'Conv{}DTranspose'):
            for num_axes in (1, 2, 3):
                x = x1 if num_axes == 1 else (x2 if num_axes == 2 else x3)
                conv_layer = getattr(tf.keras.layers, conv_type.format(num_axes), None)
                _ = conv_layer(filters=3, kernel_size=3)(x)
        _ = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding=1)(x2)
        try:
            _ = tf.keras.layers.Conv2D(3, 3, 1).build((None, 3, 3, None))
        except ValueError:
            pass
        for conv in ('Conv2D', 'Conv2DTranspose'):
            try:
                _ = getattr(tf.keras.layers, conv)(3, 3, 1, groups=2)(x2)
            except ValueError:
                pass

    def test_merging_layers(self):
        x = tf.zeros((2, 3, 3))
        _ = tf.keras.layers.Add()([x, x, x])
        _ = tf.keras.layers.Concatenate(axis=1)([x, x])
        _ = tf.keras.layers.Maximum()([x, x])
        _ = tf.keras.layers.Minimum()([x, x])
        _ = tf.keras.layers.Multiply()([x, x])
        _ = tf.keras.layers.Subtract()([x, x])
        for inputs in (x, [x], [x, x], [x, x, x], [x, x, x, x]):
            try:
                _ = tf.keras.layers.Add()(inputs)
                if len(inputs) == 1:
                    _ = tf.keras.layers.Subtract()(inputs)
                if len(inputs) == 2:
                    _ = tf.keras.layers.Subtract().call(inputs + [inputs[0]])
                if len(inputs) == 3:
                    _ = tf.keras.layers.Subtract()(inputs)
                _ = tf.keras.layers.Add().call(inputs[0])
            except ValueError:
                pass

    def test_normalization_layers(self):
        x = tf.zeros((1, 2, 3))
        _ = tf.keras.layers.BatchNormalization()(x)
        _ = tf.keras.layers.BatchNormalization(dtype=tf.float16).build(x.shape)
        _ = tf.keras.layers.LayerNormalization()(x)
        _ = tf.keras.layers.LayerNormalization(dtype=tf.float16).build(x.shape)
        for shape, axis in ((x.shape, 233), ([], 1)):
            try:
                _ = tf.keras.layers.BatchNormalization(axis=axis).build(shape)
            except ValueError:
                pass
        for shape, axis in ((x.shape, 233), ([], 1)):
            try:
                _ = tf.keras.layers.LayerNormalization(axis=axis).build(shape)
            except ValueError:
                pass

    def test_pooling_layers(self):
        x1 = tf.zeros((2, 3, 4))
        x2 = tf.zeros((2, 3, 4, 5))
        x3 = tf.zeros((2, 3, 4, 5, 6))
        _ = tf.keras.layers.AvgPool2D(pool_size=3, strides=1, padding=1)(x2)
        self.assertEqual(_.shape, x2.shape)
        for mode in ('Avg', 'Max', 'GlobalAvg', 'GlobalMax'):
            for num_axes in (1, 2, 3):
                x = x1 if num_axes == 1 else (x2 if num_axes == 2 else x3)
                _ = getattr(tf.keras.layers, '{}Pool{}D'.format(mode, num_axes))()(x)

    def test_reshaping_layers(self):
        x1 = tf.zeros((2, 3, 4))
        x2 = tf.zeros((2, 3, 4, 5))
        x3 = tf.zeros((2, 3, 4, 5, 6))
        _ = tf.keras.layers.Flatten()(x2)
        _ = tf.keras.layers.Permute((3, 1, 2))(x2)
        _ = tf.keras.layers.Reshape((-1,))(x2)
        for data_format in ('channels_last', 'channels_first'):
            for num_axes in (1, 2, 3):
                x = x1 if num_axes == 1 else (x2 if num_axes == 2 else x3)
                padding_layer = getattr(tf.keras.layers, 'ZeroPadding{}D'.format(num_axes))
                upsampling_layer = getattr(tf.keras.layers, 'UpSampling{}D'.format(num_axes))
                _ = padding_layer(data_format=data_format)(x)
                _ = upsampling_layer(data_format=data_format)(x)
        try:
            _ = tf.keras.layers.Permute((0, 3, 1, 2))(x)
        except ValueError:
            pass


if __name__ == '__main__':
    run_tests()
