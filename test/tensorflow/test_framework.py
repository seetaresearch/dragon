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
"""Test framework module."""

import unittest

from dragon.core.framework.context import get_device
from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import tensorflow as tf
import numpy as np


class TestFramework(unittest.TestCase):
    """Test framework."""

    def test_dtype(self):
        dtype_uint8 = tf.dtypes.DType("uint8")
        dtype_int32 = tf.dtypes.DType("int32")
        dtype_float32 = tf.dtypes.DType("float32")
        self.assertEqual(dtype_float32.as_numpy_dtype, np.float32)
        self.assertEqual(dtype_float32.is_numpy_compatible, True)
        self.assertEqual(dtype_float32.is_floating, True)
        self.assertEqual(dtype_int32.is_integer, True)
        self.assertEqual(dtype_int32.is_bool, False)
        self.assertEqual(dtype_uint8.is_unsigned, True)
        self.assertEqual(dtype_float32.max, np.finfo(np.float32).max)
        self.assertEqual(dtype_float32.min, np.finfo(np.float32).min)
        self.assertEqual(dtype_float32.name, "float32")
        self.assertEqual(dtype_int32.limits[1], 2**31 - 1)
        self.assertTrue(dtype_float32 != dtype_int32)
        self.assertTrue(int(dtype_int32) == dtype_int32)
        self.assertEqual(len({dtype_int32, dtype_int32}), 1)
        self.assertTrue(dtype_float32.is_compatible_with(np.float32))

    def test_ops(self):
        x = tf.convert_to_tensor(1)
        self.assertEqual(int(x), 1)
        self.assertEqual(tf.convert_to_tensor(x).id, x.id)
        with tf.name_scope("layer1"):
            x = tf.Variable(1, name="weight")
            self.assertEqual(x.name, "layer1/weight")
        for device_type in ("cpu", "gpu", "cuda", "mps"):
            with tf.device("{}:0".format(device_type)):
                if device_type == "gpu":
                    device_type = "cuda"
                self.assertEqual(get_device().type, device_type)
        for device_type in (1, "xpu:0", "gpu:none"):
            try:
                _ = tf.device(device_type)
            except (TypeError, ValueError):
                pass

    def test_tensor_shape(self):
        tensor_shape = tf.TensorShape([2, 3, 3])
        self.assertEqual(tensor_shape.dims, [2, 3, 3])
        self.assertEqual(tensor_shape.ndims, 3)
        self.assertEqual(tensor_shape.rank, 3)
        self.assertEqual(repr(tensor_shape), "TensorShape([2, 3, 3])")
        self.assertEqual(str(tensor_shape), "(2, 3, 3)")
        self.assertEqual(tensor_shape[1:].as_list(), [3, 3])
        self.assertEqual(str(tf.TensorShape([tensor_shape[0]])), "(2,)")

    def test_tensor_spec(self):
        tensor = tf.ones((2, 3, 3), dtype=tf.float32)
        tensor_spec = tf.TensorSpec((2, 3, 3), dtype=tf.float32, name="233")
        self.assertEqual(tensor_spec.dtype, "float32")
        self.assertEqual(tensor_spec.name, "233")
        self.assertEqual(tensor_spec.shape, [2, 3, 3])
        self.assertTrue(tensor_spec.is_compatible_with(tensor))
        self.assertFalse(tensor_spec.is_compatible_with(tf.zeros((2,))))
        self.assertFalse(tensor_spec.is_compatible_with(tf.zeros((2, 3, 4))))


if __name__ == "__main__":
    run_tests()
