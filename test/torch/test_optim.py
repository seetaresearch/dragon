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
"""Test optim module."""

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import torch


class TestOptimizer(unittest.TestCase):
    """Test optimizer class."""

    def test_optimizer(self):
        buffer = torch.ones(1)
        weight = torch.ones(1, requires_grad=True)
        try:
            optimizer = torch.optim.Optimizer(weight, {})
        except TypeError:
            pass
        try:
            optimizer = torch.optim.Optimizer([], {})
        except ValueError:
            pass
        optimizer = torch.optim.Optimizer([weight], {})
        try:
            optimizer.add_param_group([weight])
        except TypeError:
            pass
        try:
            optimizer.add_param_group({"params": {"param": weight}})
        except TypeError:
            pass
        try:
            optimizer.add_param_group({"params": buffer})
        except ValueError:
            pass
        try:
            optimizer.add_param_group({"params": weight})
        except ValueError:
            pass
        _ = repr(optimizer)

    def test_adam(self):
        weight = torch.ones(1, requires_grad=True)
        entries = [
            (-0.1, (0.0, 0.0), 1e-8, False),
            (0.1, (0.0, 0.0), -1e-8, False),
            (0.1, (-0.9, 0.0), 1e-8, False),
            (0.1, (0.9, -0.999), 1e-8, False),
            (0.1, (0.9, 0.999), 1e-8, False),
            (0.1, (0.9, 0.999), 1e-8, True),
        ]
        for lr, betas, eps, amsgrad in entries:
            try:
                _ = torch.optim.Adam([weight], lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)
                _ = torch.optim.AdamW([weight], lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)
            except (ValueError, NotImplementedError):
                pass

    def test_lars(self):
        weight = torch.ones(1, requires_grad=True)
        entries = [
            (-0.1, 0, 0.001),
            (0.1, -0.1, 0.001),
            (0.1, 0, -0.001),
            (0.1, 0.9, 0.001),
        ]
        for lr, momentum, trust_coef in entries:
            try:
                _ = torch.optim.LARS([weight], lr=lr, momentum=momentum, trust_coef=trust_coef)
            except ValueError:
                pass

    def test_rmsprop(self):
        weight = torch.ones(1, requires_grad=True)
        entries = [
            (-0.1, (0.0, 0.0), 1e-8, False),
            (0.1, (0.0, 0.0), -1e-8, False),
            (0.1, (-0.99, 0.0), 1e-8, False),
            (0.1, (0.99, -0.9), 1e-8, False),
            (0.1, (0.99, 0.9), 1e-8, False),
            (0.1, (0.99, 0.9), 1e-8, True),
        ]
        for lr, (alpha, momentum), eps, centered in entries:
            try:
                _ = torch.optim.RMSprop(
                    [weight],
                    lr=lr,
                    alpha=alpha,
                    eps=eps,
                    momentum=momentum,
                    centered=centered,
                )
            except ValueError:
                pass

    def test_sgd(self):
        weight = torch.ones(1, requires_grad=True)
        entries = [(-0.1, 0, False), (0.1, -0.1, False), (0.1, 0.9, True)]
        for lr, momentum, nesterov in entries:
            try:
                _ = torch.optim.SGD([weight], lr=lr, momentum=momentum, nesterov=nesterov)
            except ValueError:
                pass

    def test_step(self):
        weight1 = torch.ones(1, requires_grad=True)
        weight2 = torch.ones(1, requires_grad=True)
        optimizer = torch.optim.SGD([weight1, weight2], 0.1)
        y = weight1 + 1
        y.backward(y)
        optimizer.step()
        self.assertLessEqual(float(weight1) - 0.8, 1e-5)
        optimizer.zero_grad()
        self.assertLessEqual(float(weight1.grad) - 0.0, 1e-5)
        optimizer.zero_grad(set_to_none=True)
        self.assertEqual(weight1.grad, None)
        for i in range(3):
            y = weight1 + 1
            y.backward(y)
            optimizer.sum_grad()
        optimizer.step()
        self.assertLessEqual(float(weight1) - 0.6, 1e-5)


if __name__ == "__main__":
    run_tests()
