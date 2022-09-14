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
"""Test jit module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import torch


class TestJit(unittest.TestCase):
    """Test jit component."""

    @torch.jit.trace(example_inputs=[
        torch.Tensor(1, dtype=torch.int64),
        torch.Tensor(1, dtype=torch.int64)])
    def func1(self, a, b, **kwargs):
        return a + b

    def test_trace(self):
        @torch.jit.trace(example_inputs=[None, None])
        def func2(a, b):
            return a + b

        @torch.jit.trace
        def func3(a, b):
            return a + b

        @torch.jit.trace(example_inputs=[None])
        def func4(a, b):
            return a + b

        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        func5 = torch.jit.trace(lambda a, b: a + b)
        m = torch.jit.trace(TestModule())
        a, b = torch.tensor([1, 2]), torch.tensor([3, 4])
        self.assertEqual(self.func1(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(func2(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(func3(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(func5(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(m(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(self.func1(a, b).numpy().tolist(), [4, 6])
        try:
            func4(a, b)
        except ValueError:
            pass


if __name__ == '__main__':
    run_tests()
