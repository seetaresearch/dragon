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
"""Test device module."""

import dragon
import unittest

from dragon.core.framework import config
from dragon.core.testing.unittest.common_utils import run_tests
from dragon.core.testing.unittest.common_utils import TEST_CUDA
from dragon.core.testing.unittest.common_utils import TEST_MPS
from dragon.core.testing.unittest.common_utils import TEST_MLU


class TestCUDA(unittest.TestCase):
    """Test cuda utilities."""

    def test_stream(self):
        stream = dragon.cuda.Stream(device_index=0)
        stream.synchronize()
        dragon.cuda.synchronize()
        self.assertEqual(stream.query(), True)
        with stream.as_default():
            self.assertEqual(stream.index, dragon.get_workspace().get_stream())

    def test_cublas(self):
        dragon.cuda.set_cublas_flags()

    def test_cudnn(self):
        dragon.cuda.set_cudnn_flags()

    def test_device(self):
        count = dragon.cuda.get_device_count()
        name = dragon.cuda.get_device_name(0)
        major, _ = dragon.cuda.get_device_capability(0)
        self.assertGreaterEqual(count, 1 if TEST_CUDA else 0)
        self.assertGreaterEqual(major, 1 if TEST_CUDA else 0)
        self.assertGreaterEqual(len(name), 1 if TEST_CUDA else 0)
        dragon.cuda.set_device(0)
        self.assertEqual(dragon.cuda.current_device(), 0)
        dragon.cuda.set_default_device(1)
        self.assertEqual(config.config().device_type, "cuda")
        self.assertEqual(config.config().device_index, 1)
        dragon.cuda.set_default_device(-1)
        self.assertEqual(config.config().device_type, "cpu")
        self.assertEqual(config.config().device_index, 0)
        dragon.cuda.set_random_seed(1337)
        self.assertGreaterEqual(dragon.cuda.memory_allocated(), 0)


class TestMPS(unittest.TestCase):
    """Test mps utilities."""

    def test_stream(self):
        dragon.mps.synchronize()

    def test_device(self):
        count = dragon.mps.get_device_count()
        name = dragon.mps.get_device_name(0)
        family = dragon.mps.get_device_family(0)
        self.assertGreaterEqual(count, 1 if TEST_MPS else 0)
        self.assertGreaterEqual(len(name), 1 if TEST_MPS else 0)
        self.assertTrue("Mac2" in family if TEST_MPS else True)
        dragon.mps.set_device(0)
        self.assertEqual(dragon.mps.current_device(), 0)
        dragon.mps.set_default_device(1)
        self.assertEqual(config.config().device_type, "mps")
        self.assertEqual(config.config().device_index, 1)
        dragon.mps.set_default_device(-1)
        self.assertEqual(config.config().device_type, "cpu")
        self.assertEqual(config.config().device_index, 0)
        dragon.mps.set_random_seed(1337)
        self.assertGreaterEqual(dragon.mps.memory_allocated(), 0)


class TestMLU(unittest.TestCase):
    """Test mlu utilities."""

    def test_stream(self):
        dragon.mlu.synchronize()

    def test_device(self):
        count = dragon.mlu.get_device_count()
        name = dragon.mlu.get_device_name(0)
        major, _ = dragon.mlu.get_device_capability(0)
        self.assertGreaterEqual(count, 1 if TEST_MLU else 0)
        self.assertGreaterEqual(major, 1 if TEST_MLU else 0)
        self.assertGreaterEqual(len(name), 1 if TEST_MLU else 0)
        dragon.mlu.set_device(0)
        self.assertEqual(dragon.mlu.current_device(), 0)
        dragon.mlu.set_default_device(1)
        self.assertEqual(config.config().device_type, "mlu")
        self.assertEqual(config.config().device_index, 1)
        dragon.mlu.set_default_device(-1)
        self.assertEqual(config.config().device_type, "cpu")
        self.assertEqual(config.config().device_index, 0)
        dragon.mlu.set_random_seed(1337)
        self.assertGreaterEqual(dragon.mlu.memory_allocated(), 0)

    def test_cnnl(self):
        dragon.mlu.set_cnnl_flags()


if __name__ == "__main__":
    run_tests()
