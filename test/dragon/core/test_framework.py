# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import unittest

import dragon
import numpy as np

from dragon.core.eager.context import context as execution_context
from dragon.core.testing.unittest.common_utils import run_tests
from dragon.core.testing.unittest.common_utils import TEST_CUDA


class TestTensor(unittest.TestCase):
    """Test the tensor class."""

    def test_properties(self):
        a, b = dragon.Tensor(), dragon.EagerTensor(0)
        self.assertEqual(dragon.Tensor().ndim, 0)
        self.assertEqual(dragon.Tensor(shape=(2,)).ndim, 1)
        self.assertEqual(dragon.Tensor().shape, None)
        self.assertEqual(dragon.Tensor(shape=(2,)).shape, [2])
        self.assertEqual(dragon.Tensor().size, 0)
        self.assertEqual(dragon.Tensor(shape=(2, None)).size, math.inf)
        self.assertEqual(dragon.Tensor(shape=(2,)).size, 2)
        self.assertEqual(dragon.Tensor().dtype, None)
        self.assertEqual(dragon.Tensor(dtype='float32').dtype, 'float32')
        self.assertEqual(dragon.EagerTensor(shape=(2,)).ndim, 1)
        self.assertEqual(dragon.EagerTensor(shape=(2,)).shape, [2])
        self.assertEqual(dragon.EagerTensor(shape=(2,)).size, 2)
        self.assertEqual(dragon.EagerTensor(shape=(2,), dtype='float32').dtype, 'float32')
        self.assertEqual(dragon.EagerTensor().device, dragon.EagerTensor().device)
        self.assertNotEqual(a.__hash__(), b.__hash__())
        self.assertNotEqual(a.__repr__(), b.__repr__())
        self.assertNotEqual(b.__repr__(), dragon.EagerTensor([2]).__repr__())
        self.assertEqual(int(a.variable().placeholder().set_value(1)), 1)
        self.assertEqual(float(dragon.Tensor.convert_to(1)), 1.)
        self.assertEqual(int(b.set_value(1)), 1)
        self.assertEqual(float(b), 1.)
        self.assertEqual(int(b.get_value()), 1)
        try:
            a.shape = 1
        except TypeError:
            pass
        try:
            b.shape = (2, 3)
        except RuntimeError:
            pass
        try:
            b.dtype = 'float64'
        except RuntimeError:
            pass
        try:
            b = dragon.EagerTensor(0, 0)
        except ValueError:
            pass
        with dragon.name_scope('a'):
            a.name = 'a'
            self.assertEqual(a.name, 'a/a')
        with dragon.name_scope(''):
            b.name = 'b'
            self.assertEqual(b.name, 'b')

    def test_dlpack_converter(self):
        data = np.array([0., 1., 2.], 'float32')
        with dragon.device('cpu'), dragon.eager_scope():
            x = dragon.EagerTensor(data, copy=True)
        x_to_dlpack = dragon.dlpack.to_dlpack(x)
        x_from_dlpack = dragon.dlpack.from_dlpack(x_to_dlpack)
        self.assertEqual(x_from_dlpack.shape, list(data.shape))
        self.assertEqual(x_from_dlpack.dtype, str(data.dtype))
        self.assertLessEqual(np.abs(x_from_dlpack.numpy() - data).max(), 1e-5)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_dlpack_converter_cuda(self):
        data = np.array([0., 1., 2.], 'float32')
        with dragon.device('cuda', 0), execution_context().mode('EAGER_MODE'):
            x = dragon.EagerTensor(data, copy=True) + 0
        x_to_dlpack = dragon.dlpack.to_dlpack(x)
        x_from_dlpack = dragon.dlpack.from_dlpack(x_to_dlpack)
        self.assertEqual(x_from_dlpack.device.type, 'cuda')
        self.assertEqual(x_from_dlpack.device.index, 0)
        self.assertEqual(x_from_dlpack.shape, list(data.shape))
        self.assertEqual(x_from_dlpack.dtype, str(data.dtype))
        self.assertLessEqual(np.abs(x_from_dlpack.numpy() - data).max(), 1e-5)


class TestWorkspace(unittest.TestCase):
    """Test the workspace class."""

    def test_merge_form(self):
        w1, w2 = dragon.Workspace(), dragon.Workspace()
        with w1.as_default():
            x = dragon.Tensor(str(id(w1))).set_value(0)
        w2.merge_from(w1)
        with w2.as_default():
            self.assertEqual(int(x), 0)


if __name__ == '__main__':
    run_tests()
