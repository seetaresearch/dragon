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
"""Test the framework module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import unittest

import dragon
import numpy as np

from dragon.core.testing.unittest.common_utils import run_tests
from dragon.core.testing.unittest.common_utils import TEST_CUDA


class TestContext(unittest.TestCase):
    """Test the framework context."""

    def test_properties(self):
        dragon.random.set_seed(1337)
        dragon.set_num_threads(dragon.get_num_threads())

    def test_device(self):
        try:
            with dragon.device('abc'):
                pass
        except ValueError:
            pass

    def test_name_scope(self):
        with dragon.name_scope('a'):
            with dragon.name_scope(''):
                self.assertEqual(dragon.Tensor((), name='a').name, 'a/a')

    def test_variable_scope(self):
        with dragon.variable_scope('MyVariable'):
            x = dragon.Tensor(())
            self.assertTrue(x.id.startswith('MyVariable'))


class TestDeviceSpec(unittest.TestCase):
    """Test the device spec."""

    def test_properties(self):
        spec = dragon.DeviceSpec()
        self.assertEqual(str(spec), 'cpu:0')
        self.assertEqual(repr(spec), 'device(type=cpu, index=0)')
        self.assertNotEqual(spec, dragon.DeviceSpec('cpu', 1))


class TestGradientTape(unittest.TestCase):
    """Test the gradient tape."""

    def test_pop_push(self):
        with dragon.GradientTape() as tape:
            tape.reset()
            try:
                tape._pop_tape()
            except ValueError:
                pass
            try:
                with tape.stop_recording():
                    pass
            except ValueError:
                pass
            tape._push_tape()
            with tape.stop_recording():
                tape._tape = None
                try:
                    tape.watch(self)
                except RuntimeError:
                    pass
                self.assertEqual(tape._recording, False)
            try:
                tape._tape = None
                with tape.stop_recording():
                    pass
            except ValueError:
                pass
            try:
                tape._push_tape()
            except ValueError:
                pass


class TestTensor(unittest.TestCase):
    """Test the tensor class."""

    def test_properties(self):
        a, b = dragon.Tensor(()), dragon.Tensor(())
        self.assertEqual(dragon.Tensor(()).ndim, 0)
        self.assertEqual(dragon.Tensor(()).size, 1)
        self.assertEqual(dragon.Tensor(shape=(2,)).ndim, 1)
        self.assertEqual(dragon.Tensor(shape=(2,)).shape, (2,))
        self.assertEqual(dragon.Tensor(shape=(2,)).size, 2)
        self.assertEqual(dragon.Tensor(shape=(2,), dtype='float32').dtype, 'float32')
        self.assertEqual(dragon.Tensor(None, symbolic=True).size, 0)
        self.assertEqual(dragon.Tensor((), symbolic=True).size, 1)
        self.assertEqual(dragon.Tensor(None, symbolic=True).shape, None)
        self.assertEqual(dragon.Tensor(shape=(2, None), symbolic=True).size, math.inf)
        self.assertEqual(dragon.Tensor(None, dtype='float32', symbolic=True).dtype, 'float32')
        self.assertEqual(dragon.Tensor(None, None, symbolic=True).dtype, None)
        self.assertNotEqual(a.__hash__(), b.__hash__())
        self.assertEqual(a.__repr__(), b.__repr__())
        self.assertNotEqual(a.__repr__(), dragon.Tensor((), symbolic=True).__repr__())
        self.assertEqual(float(int(a)), float(b))
        self.assertEqual(dragon.constant([2]).item(), 2)
        self.assertEqual(dragon.constant([2, 3]).tolist(), [2, 3])
        try:
            _ = dragon.Tensor(None)
        except ValueError:
            pass
        b.requires_grad = True
        self.assertEqual(b.requires_grad, True)

    def test_dlpack_converter(self):
        data = np.array([0., 1., 2.], 'float32')
        with dragon.device('cpu'):
            x = dragon.constant(data, copy=True)
        x_to_dlpack = dragon.dlpack.to_dlpack(x)
        x_from_dlpack = dragon.dlpack.from_dlpack(x_to_dlpack)
        self.assertEqual(x_from_dlpack.shape, data.shape)
        self.assertEqual(x_from_dlpack.dtype, str(data.dtype))
        self.assertLessEqual(np.abs(x_from_dlpack.numpy() - data).max(), 1e-5)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_dlpack_converter_cuda(self):
        data = np.array([0., 1., 2.], 'float32')
        with dragon.device('cuda', 0):
            x = dragon.constant(data, copy=True) + 0
        x_to_dlpack = dragon.dlpack.to_dlpack(x)
        x_from_dlpack = dragon.dlpack.from_dlpack(x_to_dlpack)
        self.assertEqual(x_from_dlpack.device.type, 'cuda')
        self.assertEqual(x_from_dlpack.device.index, 0)
        self.assertEqual(x_from_dlpack.shape, data.shape)
        self.assertEqual(x_from_dlpack.dtype, str(data.dtype))
        self.assertLessEqual(np.abs(x_from_dlpack.numpy() - data).max(), 1e-5)


class TestWorkspace(unittest.TestCase):
    """Test the workspace class."""

    def test_clear(self):
        w = dragon.Workspace()
        with w.as_default():
            x = dragon.Tensor((1,))
        self.assertEqual(x.size, 1)
        w.clear()
        self.assertEqual(x.size, 0)

    def test_merge_form(self):
        w1, w2 = dragon.Workspace(), dragon.Workspace()
        with w1.as_default():
            x = dragon.constant(0)
        w2.merge_from(w1)
        with w2.as_default():
            self.assertEqual(int(x), 0)

    def test_register_alias(self):
        w = dragon.Workspace()
        with w.as_default():
            x = dragon.constant(1)
            w.set_alias(x.id, 'test_register_alias/y')
            alias_impl = w.get_tensor('test_register_alias/y')
            self.assertEqual(int(alias_impl.ToNumpy()), 1)

    def test_reset_workspace(self):
        w = dragon.Workspace()
        with w.as_default():
            try:
                dragon.reset_workspace()
            except AssertionError:
                pass
        dragon.reset_workspace()

    def test_memory_allocated(self):
        w = dragon.Workspace()
        with w.as_default():
            _ = w.memory_allocated()
            _ = dragon.cuda.memory_allocated()


if __name__ == '__main__':
    run_tests()
