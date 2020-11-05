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
"""Test the jit module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import torch


class TestJit(unittest.TestCase):
    """Test the jit component."""

    @torch.jit.trace(example_inputs=[
        torch.Tensor(1, dtype=torch.int64),
        torch.Tensor(1, dtype=torch.int64),
    ])
    def func1(self, a, b, **kwargs):
        """
        Return the two arguments of two arguments.

        Args:
            self: (todo): write your description
            a: (int): write your description
            b: (int): write your description
        """
        _ = kwargs
        return a + b

    def test_trace(self):
        """
        Perform a function.

        Args:
            self: (todo): write your description
        """
        @torch.jit.trace(example_inputs=[None, None])
        def func2(a, b):
            """
            Returns a and b

            Args:
                a: (int): write your description
                b: (int): write your description
            """
            return a + b

        @torch.jit.trace
        def func3(a, b):
            """
            Func3 version of b of b.

            Args:
                a: (int): write your description
                b: (int): write your description
            """
            return a + b

        @torch.jit.trace(example_inputs=[None])
        def func4(a, b):
            """
            Returns a b.

            Args:
                a: (int): write your description
                b: (int): write your description
            """
            return a + b

        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                """
                Forward a b.

                Args:
                    self: (todo): write your description
                    a: (todo): write your description
                    b: (todo): write your description
                """
                return a + b

        func5 = torch.jit.trace(lambda a, b: a + b)
        m = torch.jit.trace(TestModule())
        a, b, c = torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor(1)
        self.assertEqual(self.func1(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(func2(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(func3(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(func5(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(m(a, b).numpy().tolist(), [4, 6])
        self.assertEqual(self.func1(a, b, c=c).numpy().tolist(), [4, 6])
        try:
            func4(a, b)
        except ValueError:
            pass


if __name__ == '__main__':
    run_tests()
