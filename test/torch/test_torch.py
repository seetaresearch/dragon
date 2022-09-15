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
"""Test torch module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import io
import os.path
import tempfile
import unittest

import numpy as np

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import torch


class TestDevice(unittest.TestCase):
    """Test size class."""

    def test_properties(self):
        self.assertEqual(str(torch.device()), "cpu:0")
        self.assertEqual(repr(torch.device()), "device(type='cpu', index=0)")


class TestTensor(unittest.TestCase):
    """Test tensor class."""

    def test_properties(self):
        a = torch.tensor([0.]).cpu()
        b = torch.Tensor([0., 1.], dtype=torch.float64).zero_()
        a.requires_grad = True
        c = a + b
        c.retain_grad()
        c.backward()
        self.assertEqual(a.is_leaf, True)
        self.assertEqual(a.is_floating_point(), True)
        self.assertEqual(a.is_contiguous(), True)
        self.assertEqual(a.contiguous().is_contiguous(), True)
        self.assertEqual(a.volatile, False)
        self.assertEqual(a.numel(), 1)
        self.assertEqual(a.grad_fn, None)
        self.assertEqual(float(a.grad), 2.)
        self.assertEqual(b.grad, None)
        self.assertEqual(int(a.detach()), 0)
        self.assertEqual(torch.Tensor([0]).dim(), 1)
        self.assertEqual(float(torch.Tensor(1).one_()), 1.)
        self.assertEqual(torch.tensor(2.333).item(), 2.333)
        self.assertEqual(torch.tensor([2, 3]).tolist(), [2, 3])
        self.assertEqual(torch.empty(2, 3).ndimension(), 2)
        self.assertEqual(torch.empty(3).new_empty(2, 3).ndimension(), 2)
        self.assertEqual(repr(torch.tensor(1)), '1')
        self.assertEqual(repr(torch.tensor(1).new_tensor(1)), '1')
        self.assertNotEqual(a.__hash__(), b.__hash__())
        self.assertNotEqual(a.__repr__(), b.__repr__())
        self.assertEqual(torch.BoolTensor(1).dtype, 'bool')
        self.assertEqual(torch.ByteTensor(1).dtype, 'uint8')
        self.assertEqual(torch.CharTensor(1).dtype, 'int8')
        self.assertEqual(torch.DoubleTensor(1).dtype, 'float64')
        self.assertEqual(torch.FloatTensor(1).dtype, 'float32')
        self.assertEqual(torch.HalfTensor(1).dtype, 'float16')
        self.assertEqual(torch.IntTensor(1).dtype, 'int32')
        self.assertEqual(torch.LongTensor(1).dtype, 'int64')
        self.assertEqual(torch.autograd.Variable(torch.Tensor(1)).requires_grad, False)
        try:
            _ = torch.Tensor(5.)
        except ValueError:
            pass
        try:
            _ = torch.Tensor(2, 3.)
        except ValueError:
            pass
        try:
            torch.Tensor(2).retain_grad()
        except RuntimeError:
            pass

    def test_dlpack_converter(self):
        data = np.array([0., 1., 2.], 'float32')
        x = torch.tensor(data)
        x_to_dlpack = torch.utils.dlpack.to_dlpack(x)
        x_from_dlpack = torch.utils.dlpack.from_dlpack(x_to_dlpack)
        self.assertEqual(x_from_dlpack.shape, data.shape)
        self.assertEqual(x_from_dlpack.dtype, str(data.dtype))
        self.assertLessEqual(np.abs(x_from_dlpack.numpy() - data).max(), 1e-5)

    def test_internal_converter(self):
        data = np.array([0., 1., 2.], 'float32')
        x = torch.tensor(data)
        y = x.to(torch.int32)
        self.assertEqual(y.dtype, 'int32')
        y = x.to(torch.device('cpu'))
        self.assertEqual(y.device, torch.device('cpu'))
        y = x.to(torch.FloatTensor(1))
        self.assertEqual(y.dtype, 'float32')
        self.assertEqual(y.device, torch.device('cpu'))
        try:
            _ = x.to(data)
        except ValueError:
            pass
        try:
            _ = x.to(torch.device('gpu'))
        except ValueError:
            pass

    def test_numpy_converter(self):
        data = np.array([0., 1., 2.], 'float32')
        x = torch.from_numpy(data)
        self.assertEqual(x.shape, data.shape)
        self.assertEqual(x.dtype, str(data.dtype))
        self.assertLessEqual(np.abs(x.numpy() - data).max(), 1e-5)
        try:
            _ = torch.from_numpy(1)
        except TypeError:
            pass


class TestSize(unittest.TestCase):
    """Test size class."""

    def test_properties(self):
        self.assertEqual(torch.Size().numel(), 1)
        self.assertEqual(torch.Size((2, 3)).numel(), 6)
        self.assertEqual(repr(torch.Size()), 'torch.Size([])')


class TestSerialization(unittest.TestCase):
    """Test serialization utility."""

    def test_save_and_load(self):
        state_dict = collections.OrderedDict([
            ('a', torch.Tensor(2, 3)),
            ('b', 1),
            ('c', [1, 2, 3]),
            ('d', {'e': torch.Tensor(2), 'f': [4, 5, 6], 'g': {'h': 1}}),
        ])
        f = io.BytesIO()
        torch.save(state_dict, f)
        f.seek(0)
        state_dict2 = torch.load(f)
        self.assertEqual(state_dict['b'], state_dict2['b'])
        self.assertEqual(state_dict['c'], state_dict2['c'])
        self.assertEqual(state_dict['d']['f'], state_dict2['d']['f'])
        self.assertEqual(state_dict['d']['g']['h'], state_dict2['d']['g']['h'])
        f = io.BytesIO()
        torch.save(torch.Tensor(2, 3), f)
        f = io.BytesIO()
        torch.save([1, 2, 3], f)
        f = os.path.join(tempfile.gettempdir(), 'test_dragon_vm_torch_save')
        try:
            torch.save(state_dict, f)
            state_dict2 = torch.load(f)
            self.assertEqual(state_dict['b'], state_dict2['b'])
            self.assertEqual(state_dict['c'], state_dict2['c'])
            self.assertEqual(state_dict['d']['f'], state_dict2['d']['f'])
            self.assertEqual(state_dict['d']['g']['h'], state_dict2['d']['g']['h'])
        except (OSError, PermissionError):
            pass


if __name__ == '__main__':
    run_tests()
