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
"""Test eager module."""

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import tensorflow as tf


class TestFunction(unittest.TestCase):
    """Test function."""

    def __init__(self, method_name="runTest"):
        super(TestFunction, self).__init__(method_name)
        self.model = tf.keras.layers.ReLU()

    def test_def_function(self):
        def add(a, b):
            return a + b

        self.assertEqual(tf.function(add)(1, 2), 3)
        model_func = tf.function(self.model)
        self.assertEqual(model_func(tf.constant(-1, dtype=tf.float32)), 0)


class TestGradientTape(unittest.TestCase):
    """Test gradient tape."""

    def test_grad(self):
        x = tf.ones((2, 3))
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = x + x
        _ = tape.gradient(y, x)[0]


if __name__ == "__main__":
    run_tests()
