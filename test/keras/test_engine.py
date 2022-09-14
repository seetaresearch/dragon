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
"""Test keras engine."""

import tempfile
import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import tensorflow as tf


class TestEngine(unittest.TestCase):
    """Test keras engine."""

    def test_layer(self):
        layer = tf.keras.layers.Layer(trainable=False)
        self.assertEqual(layer.dtype, tf.float32)
        self.assertEqual(layer.trainable, False)
        self.assertEqual(layer.trainable_weights, [])
        layer.trainable = True
        self.assertEqual(layer.trainable, True)
        layer.sublayer1 = tf.keras.layers.Layer(name='sublayer1')
        layer.sublayer2 = tf.keras.layers.Layer()
        self.assertEqual(layer.sublayer1.name, 'sublayer1')
        self.assertEqual(layer.sublayer2.name, 'sublayer2')
        layer.buffer1 = tf.Variable(0, trainable=False)
        layer.weight1 = tf.Variable(1, trainable=True)
        layer.weight2, layer.buffer2 = layer.weight1, layer.buffer1
        self.assertTrue(layer.trainable_weights[0] is layer.weight1)
        self.assertTrue(layer.non_trainable_weights[0] is layer.buffer1)
        layer.trainable = False
        self.assertEqual(len(layer.weights), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)
        self.assertEqual(layer.sublayer1.trainable, False)
        self.assertEqual(layer.sublayer2.trainable, False)
        layer.input_spec = tf.keras.layers.InputSpec(shape=[3])
        try:
            layer.input_spec = [3]
        except TypeError:
            pass
        layer(tf.constant([2, 3, 3]))
        layer.save_weights(tempfile.gettempdir() + '/keras_layer_weights.pkl')
        layer.load_weights(tempfile.gettempdir() + '/keras_layer_weights.pkl')
        for suffix, save_format in (('.h5', None), ('.tf', None),
                                    ('.h5', 'h5'), ('.tf', 'tf'), ('.pth', 'pth')):
            try:
                layer.save_weights(tempfile.gettempdir() +
                                   '/keras_layer_weights{}'.format(suffix),
                                   save_format=save_format)
            except ValueError:
                pass
            if suffix is not None:
                try:
                    layer.load_weights(tempfile.gettempdir() +
                                       '/keras_layer_weights{}'.format(suffix))
                except ValueError:
                    pass

    def test_input_layer(self):
        layer1 = tf.keras.Input(shape=(3, 3), dtype='float32')
        layer2 = tf.keras.Input(batch_shape=(None, 3, 3))
        self.assertEqual(layer1.shape, layer2.shape)
        try:
            _ = tf.keras.Input(1)
            _ = tf.keras.Input(tf.TensorShape([2]))
            _ = tf.keras.Input(tensor=tf.zeros((2, 3, 3)))
            _ = tf.keras.Input()
        except ValueError:
            pass
        try:
            _ = tf.keras.Input(batch_shape=(None, 3, 3), shape=(3, 3))
        except ValueError:
            pass

    def test_input_spec(self):
        repr(tf.keras.layers.InputSpec(dtype=tf.float32))
        for axes in ('a', 99, {1: 1}, {99: 99}):
            try:
                _ = tf.keras.layers.InputSpec(shape=(2, 3), axes=axes, max_ndim=2)
            except (TypeError, ValueError):
                pass

    def test_sequential(self):
        layer = tf.keras.Sequential([tf.keras.layers.ReLU(),
                                     tf.keras.layers.ReLU()])
        layer.add(tf.keras.layers.Dropout(0.5))
        self.assertEqual(len(layer.layers), 3)
        _ = layer(tf.zeros((2, 3)), training=False)
        for i in range(3):
            layer.pop()
        try:
            layer.add(1)
        except TypeError:
            pass
        try:
            layer.pop()
        except TypeError:
            pass


if __name__ == '__main__':
    run_tests()
