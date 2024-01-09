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
"""Test backends module."""

import unittest

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.core.testing.unittest.common_utils import TEST_CUDA
from dragon.core.testing.unittest.common_utils import TEST_MLU
from dragon.core.testing.unittest.common_utils import TEST_MPS
from dragon.vm import torch


class TestCUDA(unittest.TestCase):
    """Test CUDA backend."""

    def test_library(self):
        _ = torch.backends.cuda.is_built()

    def test_set_flags(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, False)

    def test_device(self):
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        major, _ = torch.cuda.get_device_capability(0)
        self.assertGreaterEqual(count, 1 if TEST_CUDA else 0)
        self.assertGreaterEqual(major, 1 if TEST_CUDA else 0)
        self.assertGreaterEqual(len(name), 1 if TEST_CUDA else 0)
        torch.cuda.set_device(0)
        self.assertEqual(torch.cuda.current_device(), 0)
        torch.cuda.synchronize()
        torch.cuda.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)
        self.assertGreaterEqual(torch.cuda.memory_allocated(), 0)

    def test_graph(self):
        for graph_type in (torch.cuda.TraceGraph, torch.cuda.CUDAGraph):
            with torch.cuda.graph(graph_type()) as builder:
                pass
            builder.cuda_graph.replay()
            builder.cuda_graph.reset()


class TestCuDNN(unittest.TestCase):
    """Test CuDNN backend."""

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


class TestMPS(unittest.TestCase):
    """Test MPS backend."""

    def test_library(self):
        _ = torch.backends.mps.is_built()
        if torch.backends.mps.is_available():
            self.assertEqual(torch.backends.mps.is_built(), True)

    def test_device(self):
        count = torch.mps.device_count()
        name = torch.mps.get_device_name(0)
        families = torch.mps.get_device_family(0)
        self.assertGreaterEqual(count, 1 if TEST_MPS else 0)
        self.assertGreaterEqual(len(families), 1 if TEST_MPS else 0)
        self.assertGreaterEqual(len(name), 1 if TEST_MPS else 0)
        torch.mps.set_device(0)
        self.assertEqual(torch.mps.current_device(), 0)
        torch.mps.synchronize()
        torch.mps.manual_seed(1337)
        torch.mps.manual_seed_all(1337)
        self.assertGreaterEqual(torch.mps.memory_allocated(), 0)


class TestOpenMP(unittest.TestCase):
    """Test OpenMP backend."""

    def test_library(self):
        _ = torch.backends.openmp.is_available()


class TestMLU(unittest.TestCase):
    """Test MLU backend."""

    def test_library(self):
        _ = torch.backends.mlu.is_available()

    def test_device(self):
        count = torch.mlu.device_count()
        name = torch.mlu.get_device_name(0)
        major, _ = torch.mlu.get_device_capability(0)
        self.assertGreaterEqual(count, 1 if TEST_MLU else 0)
        self.assertGreaterEqual(major, 1 if TEST_MLU else 0)
        self.assertGreaterEqual(len(name), 1 if TEST_MLU else 0)
        torch.mlu.set_device(0)
        self.assertEqual(torch.mlu.current_device(), 0)
        torch.mlu.synchronize()
        torch.mlu.manual_seed(1337)
        torch.mlu.manual_seed_all(1337)
        self.assertGreaterEqual(torch.mlu.memory_allocated(), 0)


class TestCNNL(unittest.TestCase):
    """Test CNNL backend."""

    def test_library(self):
        if torch.backends.cnnl.is_available():
            self.assertGreater(torch.backends.cnnl.version(), 0)
        else:
            self.assertEqual(torch.backends.cnnl.version(), None)

    def test_set_flags(self):
        torch.backends.cnnl.enabled = True
        self.assertEqual(torch.backends.cnnl.enabled, True)


if __name__ == "__main__":
    run_tests()
