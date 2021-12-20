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
"""Test the backends module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import torch


class TestCuDNN(unittest.TestCase):
    """Test the CuDNN backend."""

    def test_library(self):
        if torch.backends.cudnn.is_available():
            self.assertGreater(torch.backends.cudnn.version(), 0)
        else:
            self.assertEqual(torch.backends.cudnn.version(), None)

    def test_set_flags(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = False
        self.assertEqual(torch.backends.cudnn.enabled, True)
        self.assertEqual(torch.backends.cudnn.benchmark, False)
        self.assertEqual(torch.backends.cudnn.deterministic, False)
        self.assertEqual(torch.backends.cudnn.allow_tf32, False)


if __name__ == '__main__':
    run_tests()
