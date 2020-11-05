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
"""Test the autograd module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import torch


class TestGradMode(unittest.TestCase):
    """Test the grad mode."""

    def test_set_grad_enabled(self):
        """
        Set the gradient of the gradient.

        Args:
            self: (todo): write your description
        """
        a = torch.tensor(1., requires_grad=True)
        b = torch.tensor(1., requires_grad=False)
        with torch.no_grad():
            self.assertEqual((a + 1).requires_grad, False)
            self.assertEqual((b + 1).requires_grad, False)
        with torch.enable_grad():
            self.assertEqual((a + 1).requires_grad, True)
            self.assertEqual((b + 1).requires_grad, False)
        with torch.set_grad_enabled(False):
            self.assertEqual((a + 1).requires_grad, False)
            self.assertEqual((b + 1).requires_grad, False)
        with torch.set_grad_enabled(True):
            self.assertEqual((a + 1).requires_grad, True)
            self.assertEqual((b + 1).requires_grad, False)


class TestBackProp(unittest.TestCase):
    """Test the back-propagation."""

    def test_backward(self):
        """
        Test for backward backward backward.

        Args:
            self: (todo): write your description
        """
        x = torch.tensor(1., dtype=torch.float32, requires_grad=True)
        y = x + 1
        entries = [([y], []),
                   ([y], [1]),
                   ([torch.tensor(1.)], []),
                   ([y], [torch.tensor([1., 1.])]),
                   ([y], [y])]
        for tensors, grad_tensors in entries:
            try:
                torch.autograd.backward(tensors, grad_tensors)
                self.assertLessEqual(float(x.grad) - 2., 1e-5)
            except (ValueError, TypeError):
                pass


if __name__ == '__main__':
    run_tests()
