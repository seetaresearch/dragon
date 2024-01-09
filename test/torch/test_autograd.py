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
"""Test autograd module."""

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import torch


class TestGradMode(unittest.TestCase):
    """Test grad mode."""

    def test_set_grad_enabled(self):
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=False)
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
        with torch.inference_mode(True):
            self.assertEqual((a + 1).requires_grad, False)
            self.assertEqual((b + 1).requires_grad, False)
        with torch.inference_mode(False):
            self.assertEqual((a + 1).requires_grad, True)
            self.assertEqual((b + 1).requires_grad, False)


class TestBackProp(unittest.TestCase):
    """Test back-propagation."""

    def test_backward(self):
        x = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
        y = x + 1
        entries = [
            ([y], []),
            ([y], [1]),
            ([torch.tensor(1.0)], []),
            ([y], [torch.tensor([1.0, 1.0])]),
            ([y], [y]),
        ]
        for tensors, grad_tensors in entries:
            try:
                torch.autograd.backward(tensors, grad_tensors)
                self.assertLessEqual(float(x.grad) - 2.0, 1e-5)
            except (ValueError, TypeError):
                pass


if __name__ == "__main__":
    run_tests()
