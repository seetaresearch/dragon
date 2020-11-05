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
"""Test the device module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon
import unittest

from dragon.core.framework import config
from dragon.core.testing.unittest.common_utils import run_tests
from dragon.core.testing.unittest.common_utils import TEST_CUDA


class TestCUDA(unittest.TestCase):
    """Test the cuda utilities."""

    def test_stream(self):
        """
        Test for the stream.

        Args:
            self: (todo): write your description
        """
        stream = dragon.cuda.Stream(device_index=0)
        self.assertGreater(stream.ptr, 0 if TEST_CUDA else -1)
        stream.synchronize()
        dragon.cuda.synchronize()

    def test_cudnn(self):
        """
        Enable test test test test.

        Args:
            self: (todo): write your description
        """
        dragon.cuda.enable_cudnn()

    def test_device(self):
        """
        Assign the device.

        Args:
            self: (todo): write your description
        """
        major, minor = dragon.cuda.get_device_capability(0)
        self.assertGreaterEqual(major, 1 if TEST_CUDA else 0)
        self.assertGreaterEqual(minor, 0)
        dragon.cuda.set_device(0)
        self.assertEqual(dragon.cuda.current_device(), 0)
        dragon.cuda.set_default_device(1)
        self.assertEqual(config.config().device_type, 'cuda')
        self.assertEqual(config.config().device_index, 1)
        dragon.cuda.set_default_device(-1)
        self.assertEqual(config.config().device_type, 'cpu')
        self.assertEqual(config.config().device_index, 0)


if __name__ == '__main__':
    run_tests()
