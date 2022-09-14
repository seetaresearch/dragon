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
"""Test device module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.core.testing.unittest.common_utils import TEST_CUDA
from dragon.vm import torch


class TestCUDA(unittest.TestCase):
    """Test cuda device."""

    def test_device(self):
        name = torch.cuda.get_device_name(0)
        major, _ = torch.cuda.get_device_capability(0)
        self.assertGreaterEqual(major, 1 if TEST_CUDA else 0)
        self.assertGreaterEqual(len(name), 1 if TEST_CUDA else 0)
        torch.cuda.set_device(0)
        self.assertEqual(torch.cuda.current_device(), 0)
        torch.cuda.synchronize()


if __name__ == '__main__':
    run_tests()
