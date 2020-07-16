# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Test the autograph module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import dragon
from dragon.core.testing.unittest.common_utils import run_tests


class TestFunction(unittest.TestCase):
    """Test the graph function."""

    @dragon.function(input_signature=[
        dragon.Tensor(dtype='int32'),
        dragon.Tensor(dtype='int32'),
        dragon.Tensor(dtype='int32'),
    ])
    def func1(self, a, b, c=0, **kwargs):
        _ = kwargs
        return a + b + c

    def test_def_function(self):
        @dragon.function(input_signature=[dragon.Tensor()])
        def func2(a, b):
            return a + b
        self.assertEqual(self.func1([1, 2], [3, 4]).get_value().tolist(), [4, 6])
        self.assertEqual(self.func1([1, 2], b=[3, 4]).get_value().tolist(), [4, 6])
        self.assertEqual(self.func1([1, 2], b=[3, 4], c=1).get_value().tolist(), [5, 7])
        self.assertEqual(self.func1([1, 2], b=[3, 4], c=1).get_value().tolist(), [5, 7])
        self.assertEqual(self.func1([1, 2], [3, 4], executing_stage='forward').get_value().tolist(), [4, 6])
        dragon.function(func=lambda: dragon.optimizers.SGD())()
        try:
            self.func1(1, 2, 3, 4)
        except ValueError:
            pass
        try:
            func2(1, 2)
        except ValueError:
            pass

    def test_update_function(self):
        optimizer = dragon.optimizers.SGD()
        try:
            _ = optimizer.op_type
        except KeyError:
            pass
        value = dragon.Tensor(dtype='float32').set_value(1.)
        grad = dragon.Tensor(dtype='float32').set_value(1.)
        optimizer.apply_gradients([(value, grad)])
        dragon.create_function(optimizer=optimizer)()


if __name__ == '__main__':
    run_tests()
