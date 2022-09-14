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
"""Test module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import tensorflow as tf


class TestModule(unittest.TestCase):
    """Test array ops."""

    def test_properties(self):
        self.assertEqual(tf.Module(name='module1').name, 'module1')
        self.assertEqual(tf.Module().name, 'module')
        m = tf.Module()
        m.module = tf.Module()
        m.weight1 = tf.Variable(1, trainable=True)
        m.buffer1 = tf.Variable(1, trainable=False)
        self.assertEqual(len(m.variables), 2)
        self.assertEqual(len(m.trainable_variables), 1)
        self.assertEqual(len(m.submodules), 1)
        with m.name_scope:
            self.assertEqual(tf.Variable(1, name='weight').name, 'module_1/weight')


if __name__ == '__main__':
    run_tests()
