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
"""Test module."""

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import tensorflow as tf


class TestModule(unittest.TestCase):
    """Test array ops."""

    def test_properties(self):
        self.assertEqual(tf.Module(name="module1").name, "module1")
        self.assertEqual(tf.Module().name, "module")
        m = tf.Module()
        m.module = tf.Module()
        m.weight1 = tf.Variable(1, trainable=True)
        m.buffer1 = tf.Variable(1, trainable=False)
        self.assertEqual(len(m.variables), 2)
        self.assertEqual(len(m.trainable_variables), 1)
        self.assertEqual(len(m.submodules), 1)
        with m.name_scope:
            self.assertEqual(tf.Variable(1, name="weight").name, "module_1/weight")


if __name__ == "__main__":
    run_tests()
