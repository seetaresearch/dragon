# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Test nn module."""

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import tensorflow as tf


class TestFunctions(unittest.TestCase):
    """Test functions."""

    def test_activations(self):
        x = tf.zeros((2, 3, 3))
        _ = tf.keras.activations.elu(x)
        _ = tf.keras.activations.exponential(x)
        _ = tf.keras.activations.hard_sigmoid(x)
        _ = tf.keras.activations.linear(x)
        _ = tf.keras.activations.relu(x)
        _ = tf.keras.activations.relu(x, max_value=6)
        _ = tf.keras.activations.selu(x)
        _ = tf.keras.activations.sigmoid(x)
        _ = tf.keras.activations.softmax(x)
        _ = tf.keras.activations.swish(x)
        _ = tf.keras.activations.tanh(x)
        try:
            _ = tf.keras.activations.relu(x, max_value=1)
        except ValueError:
            pass
        try:
            _ = tf.keras.activations.relu(x, max_value=6, alpha=0.1)
        except ValueError:
            pass
        try:
            _ = tf.keras.activations.get(1)
        except TypeError:
            pass

    def test_initializers(self):
        _ = tf.keras.initializers.get(None)
        try:
            _ = tf.keras.initializers.get(1)
        except TypeError:
            pass

    def test_losses(self):
        x = tf.zeros((2, 3, 3))
        y, y2 = tf.zeros((2, 3, 3)), tf.zeros((2, 3), dtype=tf.int64)
        _ = tf.keras.losses.BinaryCrossentropy()(y, x)
        _ = tf.keras.losses.CategoricalCrossentropy()(y, x)
        _ = tf.keras.losses.MeanAbsoluteError()(y, x)
        _ = tf.keras.losses.MeanSquaredError()(y, x)
        _ = tf.keras.losses.SparseCategoricalCrossentropy()(y2, x)
        _ = tf.keras.losses.get(None)
        _ = tf.keras.losses.get(tf.keras.losses.mean_absolute_error)
        _ = tf.keras.losses.get("mean_squared_error")
        try:
            _ = tf.keras.losses.get(1)
        except TypeError:
            pass


if __name__ == "__main__":
    run_tests()
