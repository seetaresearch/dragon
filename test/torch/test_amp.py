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
"""Test amp module."""

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import torch


class TestGradScaler(unittest.TestCase):
    """Test grad scaler."""

    def test_properties(self):
        scaler = torch.cuda.amp.GradScaler(1024)
        scaler.set_backoff_factor(0.2333)
        scaler.set_growth_factor(2.333)
        scaler.set_growth_interval(2333)
        self.assertEqual(scaler.get_backoff_factor(), 0.2333)
        self.assertEqual(scaler.get_growth_factor(), 2.333)
        self.assertEqual(scaler.get_growth_interval(), 2333)
        self.assertEqual(scaler.get_scale(), 1024)
        scaler.update(2333)
        self.assertEqual(scaler.get_scale(), 2333)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        scaler.update(), scaler.step(torch.optim.SGD([torch.tensor(1)], 1))
        self.assertEqual(scaler.get_scale(), 1.0)
        self.assertEqual(scaler.is_enabled(), False)

    def test_scale(self):
        scaler = torch.cuda.amp.GradScaler(16)
        x = torch.tensor(1.0, dtype=torch.float32)
        scaler.scale(x)
        self.assertEqual(float(x), 16.0)
        scaler.scale([x, x])
        self.assertEqual(float(x), 16.0 * 16.0 * 16.0)
        self.assertEqual(id(scaler.scale(scaler)), id(scaler))

    def test_step(self):
        scaler = torch.cuda.amp.GradScaler(1.0)
        x = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.SGD([x], lr=1.0)
        x.mul(float("inf")).backward()
        scaler.step(optimizer), self.assertEqual(float(x), 1.0)
        x.mul(float("nan")).backward()
        scaler.step(optimizer), self.assertEqual(float(x), 1.0)
        x.mul(1.0).backward()
        scaler.step(optimizer), self.assertEqual(float(x), 0.0)

    def test_update(self):
        scaler = torch.cuda.amp.GradScaler(1024, growth_interval=1)
        scaler._found_inf_value = 1.0
        scaler.update()
        self.assertEqual(scaler.get_scale(), 512.0)
        scaler._found_inf_value = 0.0
        scaler.update()
        self.assertEqual(scaler.get_scale(), 1024.0)
        scaler.update(torch.tensor(2333, dtype=torch.float32))
        self.assertEqual(scaler.get_scale(), 2333.0)


if __name__ == "__main__":
    run_tests()
