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
"""Test the autograph module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import dragon
from dragon.core.framework import config
from dragon.core.testing.unittest.common_utils import run_tests


class TestConfig(unittest.TestCase):
    """Test the graph config."""

    def test_execution(self):
        for mode in ('EAGER_MODE', 'GRAPH_MODE', 'UNKNOWN'):
            try:
                dragon.autograph.set_execution(mode)
                self.assertEqual(config.config().graph_execution, mode)
            except ValueError:
                pass

    def test_optimization(self):
        dragon.autograph.set_optimization(1)
        self.assertEqual(config.config().graph_optimization, 1)

    def test_scheduler(self):
        for scheduler in ('SIMPLE', 'FUSION', 'KNOWN', 'SIMPLE'):
            try:
                dragon.autograph.set_scheduler(scheduler)
                if scheduler == 'FUSION':
                    self.assertEqual(config.config().graph_type, 'FusionGraph')
                else:
                    self.assertEqual(config.config().graph_type, '')
            except ValueError:

                pass

    def test_verbosity(self):
        dragon.autograph.set_verbosity(1)
        self.assertEqual(config.config().graph_verbosity, 1)
        dragon.autograph.set_verbosity(0)


class TestFunction(unittest.TestCase):
    """Test the graph function."""

    @dragon.function(input_signature=[
        dragon.Tensor((1,), dtype='int32'),
        dragon.Tensor((1,), dtype='int32'),
        dragon.Tensor((1,), dtype='int32'),
    ])
    def func1(self, a, b, c=0, **kwargs):
        _ = kwargs
        return a + b + c

    def test_create_function(self):
        a = dragon.Tensor((), dtype='int32').set_value(1)
        b = dragon.Tensor((), dtype='int32').set_value(2)
        y = a + 1
        try:
            dragon.create_function(outputs=y, optimizer=dragon.optimizers.SGD())
        except ValueError:
            pass
        try:
            dragon.create_function(outputs=dragon.EagerTensor(1))
        except ValueError:
            pass
        try:
            f = dragon.create_function(outputs=y, givens={a: 1})
        except ValueError:
            f = dragon.create_function(outputs=y, givens={a: b})
        self.assertEqual(int(f()), 3)

    def test_def_function(self):
        @dragon.function(input_signature=[dragon.Tensor(None)])
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
        value = dragon.Tensor((), dtype='float32').set_value(1.)
        grad = dragon.Tensor((), dtype='float32').set_value(1.)
        optimizer.apply_gradients([(value, grad)])
        dragon.create_function(optimizer=optimizer)()


if __name__ == '__main__':
    run_tests()
