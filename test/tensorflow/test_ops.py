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
"""Test ops module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import unittest

import numpy as np

from dragon.core.util import nest
from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import tensorflow as tf

# Fix the duplicate linked omp runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Fix the numpy seed
np.random.seed(1337)


class OpTestCase(unittest.TestCase):
    """The base test case."""

    precision = 1e-5

    def __init__(self, method_name='runTest'):
        super(OpTestCase, self).__init__(method_name)

    def assertEqual(
        self,
        first,
        second,
        msg=None,
        prec=None,
    ):
        if prec is None:
            prec = self.precision
        inputs = nest.flatten(first)
        num_first = len(inputs)
        inputs += nest.flatten(second)
        num_second = len(inputs) - num_first
        for i, x in enumerate(inputs):
            inputs[i] = x.numpy() if hasattr(x, 'numpy') else x
        first = inputs[:num_first] if num_first > 1 else inputs[0]
        second = inputs[num_first:len(inputs)] if num_second > 1 else inputs[num_first]
        if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            super(OpTestCase, self).assertEqual(first.shape, second.shape)
            if first.dtype == bool and second.dtype == bool:
                diff = first ^ second
                num_unique = len(np.unique(diff))
                self.assertLessEqual(num_unique, 1, msg)
            else:
                diff = np.abs(first - second)
                max_err = diff.max()
                self.assertLessEqual(max_err, prec, msg)
        elif nest.is_sequence(first) and nest.is_sequence(second):
            for a, b in zip(first, second):
                self.assertEqual(a, b, msg, prec)
        else:
            super(OpTestCase, self).assertEqual(first, second, msg)


class TestArrayOps(OpTestCase):
    """Test array ops."""

    def test_broadcast_to(self):
        entries = [(2, 2, 3, 1),
                   (1, 2, 3, 2),
                   (2, 2, 3, 2),
                   (2, 1, 2, 3, 1)]
        for shape in entries:
            data = np.arange(6).astype('float32').reshape((1, 2, 3, 1))
            x = new_tensor(data)
            y = tf.broadcast_to(x, shape)
            self.assertEqual(y, np.broadcast_to(data, shape))

    def test_concat(self):
        entries = [0, 1]
        for axis in entries:
            data = arange((2, 2))
            x = new_tensor(data)
            y = tf.concat([x, x], axis=axis)
            self.assertEqual(y, np.concatenate([data, data], axis=axis))

    def test_depth_to_space(self):
        n, co, si = 2, 2, 2
        entries = [(2, 2, 'NCHW'), (2, 3, 'NCHW'), (2, 2, 'NHWC'), (2, 3, 'NHWC')]
        for bs, num_axes, data_format in entries:
            ci = co * int(math.pow(bs, num_axes))
            perm = [0] * (num_axes * 2 + 1) + [num_axes * 2 + 1]
            if data_format == 'NCHW':
                data1 = arange([n, ci] + [si] * num_axes)
                perm[1] = num_axes + 1
                for i in range(num_axes):
                    perm[i * 2 + 2] = num_axes + i + 2
                    perm[i * 2 + 3] = i + 1
                data2 = data1.reshape([n] + [bs] * num_axes + [co] + [si] * num_axes)
                data2 = data2.transpose(perm)
                data2 = data2.reshape([n, co] + [bs * si] * num_axes)
            else:
                data1 = arange([n] + [si] * num_axes + [ci])
                for i in range(num_axes):
                    perm[i * 2 + 1] = i + 1
                    perm[i * 2 + 2] = num_axes + i + 1
                data2 = data1.reshape([n] + [si] * num_axes + [bs] * num_axes + [co])
                data2 = data2.transpose(perm)
                data2 = data2.reshape([n] + [bs * si] * num_axes + [co])
            x = new_tensor(data1)
            y = tf.nn.depth_to_space(x, bs, data_format=data_format)
            self.assertEqual(y, data2)

    def test_expand_dims(self):
        entries = [1, -1]
        for axis in entries:
            data = arange((2, 3, 4))
            x = new_tensor(data)
            y = tf.expand_dims(x, axis=axis)
            self.assertEqual(y, np.expand_dims(data, axis=axis))

    def test_fill(self):
        entries = [((2, 3), 1), ((2, 3), 0.)]
        for shape, value in entries:
            data = np.zeros(shape, dtype='float32')
            data.fill(value)
            x = tf.fill(shape, value=value, dtype=tf.float32)
            self.assertEqual(x, data)
        self.assertEqual(tf.fill((1,), 0).dtype, 'int32')
        self.assertEqual(tf.fill((1,), 0.).dtype, 'float32')

    def test_gather(self):
        entries = [1, 2]
        for axis in entries:
            data = arange((1, 2, 3, 4))
            index = np.array([0, 1, 1], 'int64')
            x = new_tensor(data)
            x_index = new_tensor(index)
            y = tf.gather(x, x_index, axis=axis)
            self.assertEqual(y, np.take(data, index, axis=axis))

    def test_identity(self):
        data = arange((4,))
        x = new_tensor(data)
        self.assertEqual(tf.identity(x), data)

    def test_ones(self):
        x = tf.ones((2, 3), tf.float32)
        self.assertEqual(x, np.ones((2, 3), 'float32'))

    def test_ones_like(self):
        data = np.ones((2, 3), 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.ones_like(x, tf.float32), data)

    def test_one_hot(self):
        entries = [(2, 3, 3), (1, 2, 3), (2, 2, 2)]
        for index in entries:
            index = np.array(index, 'int64')
            x = new_tensor(index)
            y = tf.one_hot(x, depth=10)
            self.assertEqual(y, np.eye(10, dtype='int64')[index])

    def test_pad(self):
        entries = [([(0, 1)], 'constant'),
                   ([(1, 1)], 'reflect')]
        for pads, mode in entries:
            data = arange((6,))
            x = new_tensor(data)
            y = tf.pad(x, pads, mode.upper())
            self.assertEqual(y, np.pad(data, pads, mode))

    def test_placeholder(self):
        x = tf.placeholder(tf.float32, (2, 3))
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x._is_variable, False)

    def test_reshape(self):
        entries = [(3, -1), (2, -1)]
        for shape in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = tf.reshape(x, shape)
            self.assertEqual(y, data.reshape(y.shape))
            tf.reshape(x, shape, copy=False)
            self.assertEqual(x, data.reshape(y.shape))

    def test_reverse(self):
        entries = [0, 1, (1, 2)]
        for axis in entries:
            data = arange((2, 3, 4))
            x = new_tensor(data)
            y = tf.reverse(x, axis)
            self.assertEqual(y, np.flip(data, axis))

    def test_roll(self):
        entries = [(0, 0), ((0, 0), (0, 1)), ((-1, 1), (0, 1)), (1, None)]
        for shift, axis in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = tf.roll(x, shift, axis)
            self.assertEqual(y, np.roll(data, shift, axis))

    def test_shape(self):
        entries = [(2, 3), (2, 3, 3)]
        for shape in entries:
            self.assertEqual(tf.shape(tf.ones(shape)),
                             np.array(shape, 'int64'))

    def test_slice(self):
        entries = [0,
                   slice(0, None, None),
                   (slice(None, None, None), slice(1, None, None)),
                   (slice(None, None, None),) * 3]
        for item in entries:
            data = arange((1, 2, 3))
            grad = np.zeros_like(data, 'float32')
            grad.__setitem__(item, 1.)
            grad *= data
            x = new_tensor(data)
            y = tf.slice(x, *process_index(item))
            self.assertEqual(y, data.__getitem__(item))

    def test_space_to_depth(self):
        n, ci, so = 2, 2, 2
        entries = [(2, 2, 'NCHW'), (2, 3, 'NCHW'), (2, 2, 'NHWC'), (2, 3, 'NHWC')]
        for bs, num_axes, data_format in entries:
            co, si = ci * int(math.pow(bs, num_axes)), so * bs
            perm = [0] * (num_axes * 2 + 1) + [num_axes * 2 + 1]
            start_axis = 2 if data_format == 'NCHW' else 1
            end_axis = num_axes + 2 if data_format == 'NCHW' else num_axes + 1
            perm_count = 0
            for i in range(num_axes + 2):
                if i < start_axis:
                    perm[i] = perm_count
                    perm_count += 1
                elif start_axis <= i < end_axis:
                    perm[i] = perm_count
                    perm_count += 1
                    perm[i + num_axes] = perm_count
                    perm_count += 1
                else:
                    perm[perm_count] = perm_count
            if data_format == 'NCHW':
                for i in range(num_axes):
                    perm.insert(1, perm[-1])
                    perm.pop(-1)
                data1 = arange([n, ci] + [si] * num_axes)
                data2 = data1.reshape([n] + [ci] + [so, bs] * num_axes)
                data2 = data2.transpose(perm)
                data2 = data2.reshape([n, co] + [so] * num_axes)
            else:
                data1 = arange([n] + [si] * num_axes + [ci])
                data2 = data1.reshape([n] + [so, bs] * num_axes + [ci])
                data2 = data2.transpose(perm)
                data2 = data2.reshape([n] + [so] * num_axes + [co])
            x = new_tensor(data1)
            y = tf.nn.space_to_depth(x, bs, data_format=data_format)
            self.assertEqual(y, data2)

    def test_split(self):
        entries = [(2, 0), (2, 1), ((1, 1), 1)]
        for num, axis in entries:
            data = arange((2, 2))
            x = new_tensor(data)
            y = tf.split(x, num, axis=axis)
            self.assertEqual(y, np.split(data, (1,), axis=axis))

    def test_squeeze(self):
        entries = [((2, 1, 3), 1), ((1, 2, 1, 3), (0, 2)), ((3, 1, 2, 1), (1,))]
        for shape, axis in entries:
            data = arange(shape)
            x = new_tensor(data)
            y = tf.squeeze(x, axis)
            self.assertEqual(y, np.squeeze(data, axis))
            tf.squeeze(x, axis, copy=False)
            self.assertEqual(x, np.squeeze(data, axis))

    def test_tile(self):
        entries = [(2,), (1, 1), (1, 2), (2, 1), (2, 2)]
        for repeats in entries:
            data = arange((2, 2))
            x = new_tensor(data)
            y = tf.tile(x, repeats)
            repeats = (1,) * (len(data.shape) - len(repeats)) + repeats
            self.assertEqual(y, np.tile(data, repeats))

    def test_transpose(self):
        entries = [(0, 2, 1), None]
        for perm in entries:
            data = arange((2, 3, 4))
            x = new_tensor(data)
            y = tf.transpose(x, perm)
            self.assertEqual(y, np.transpose(data, perm))

    def test_unique(self):
        data = np.array([1, 1, 3, 5, 5, 7, 9])
        x = new_tensor(data)
        y = tf.unique(x)
        result = np.unique(data, return_inverse=True)
        self.assertEqual(y, result)

    def test_unique_with_counts(self):
        data = np.array([1, 1, 3, 5, 5, 7, 9])
        x = new_tensor(data)
        y = tf.unique_with_counts(x)
        result = np.unique(data, return_inverse=True, return_counts=True)
        self.assertEqual(y, result)

    def test_unstack(self):
        entries = [0, 1]
        for axis in entries:
            data = arange((2, 3))
            num = data.shape[axis]
            x = new_tensor(data)
            y = tf.unstack(x, axis=axis)
            result = [x.squeeze(axis) for x in np.split(data, num, axis)]
            self.assertEqual(y, result)

    def test_zeros(self):
        x = tf.zeros((2, 3), tf.float32)
        self.assertEqual(x, np.zeros((2, 3), 'float32'))

    def test_zeros_like(self):
        data = np.zeros((2, 3), 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.zeros_like(x, tf.float32), data)


class TestBitwiseOps(OpTestCase):
    """Test bitwise ops."""

    def test_bitwise_and(self):
        data1 = arange(8, dtype='uint8')
        data2 = arange(8, start=-4, dtype='uint8')
        x1, x2 = new_tensor(data1), new_tensor(data2)
        y = tf.bitwise.bitwise_and(x1, x2)
        self.assertEqual(y, np.bitwise_and(data1, data2))

    def test_bitwise_or(self):
        data1 = arange(8, dtype='uint8')
        data2 = arange(8, start=-4, dtype='uint8')
        x1, x2 = new_tensor(data1), new_tensor(data2)
        y = tf.bitwise.bitwise_or(x1, x2)
        self.assertEqual(y, np.bitwise_or(data1, data2))

    def test_bitwise_xor(self):
        data1 = arange(8, dtype='uint8')
        data2 = arange(8, start=-4, dtype='uint8')
        x1, x2 = new_tensor(data1), new_tensor(data2)
        y = tf.bitwise.bitwise_xor(x1, x2)
        self.assertEqual(y, np.bitwise_xor(data1, data2))

    def test_invert(self):
        data = arange(8, start=-4, dtype='uint8')
        x = new_tensor(data)
        y = tf.bitwise.invert(x)
        self.assertEqual(y, ~data)


class TestClipOps(OpTestCase):
    """Test clip ops."""

    def test_clip_by_value(self):
        entries = [(None, None), (2, None), (None, 4), (2, 4)]
        for low, high in entries:
            data = arange((6,))
            x = new_tensor(data)
            y = tf.clip_by_value(x, low, high)
            result = np.clip(data, low, high) if low or high else data
            self.assertEqual(y, result)


class TestInitOps(OpTestCase):
    """Test init ops."""

    def test_constant_initializer(self):
        initializer = tf.initializers.Constant(value=1, dtype=tf.float32)
        self.assertEqual(initializer((2, 3)), np.ones((2, 3), 'float32'))
        self.assertEqual(initializer((2, 3)).dtype, 'float32')
        self.assertEqual(initializer((2, 3), tf.float64).dtype, 'float64')

    def test_glorot_normal_initializer(self):
        initializer = tf.initializers.GlorotNormal(dtype=tf.float32)
        self.assertEqual(initializer((2, 3)).dtype, 'float32')
        self.assertEqual(initializer((2, 3), tf.float64).dtype, 'float64')

    def test_glorot_uniform_initializer(self):
        initializer = tf.initializers.GlorotUniform(dtype=tf.float32)
        self.assertEqual(initializer((2, 3)).dtype, 'float32')
        self.assertEqual(initializer((2, 3), tf.float64).dtype, 'float64')

    def test_ones_initializer(self):
        initializer = tf.initializers.Ones(dtype=tf.float32)
        self.assertEqual(initializer((2, 3)), np.ones((2, 3), 'float32'))
        self.assertEqual(initializer((2, 3)).dtype, 'float32')
        self.assertEqual(initializer((2, 3), tf.float64).dtype, 'float64')

    def test_random_normal_initializer(self):
        initializer = tf.initializers.RandomNormal(dtype=tf.float32)
        self.assertEqual(initializer((2, 3)).dtype, 'float32')
        self.assertEqual(initializer((2, 3), tf.float64).dtype, 'float64')

    def test_random_uniform_initializer(self):
        initializer = tf.initializers.RandomUniform(dtype=tf.float32)
        self.assertEqual(initializer((2, 3)).dtype, 'float32')
        self.assertEqual(initializer((2, 3), tf.float64).dtype, 'float64')

    def test_truncated_normal_initializer(self):
        initializer = tf.initializers.TruncatedNormal(dtype=tf.float32)
        self.assertEqual(initializer((2, 3)).dtype, 'float32')
        self.assertEqual(initializer((2, 3), tf.float64).dtype, 'float64')

    def test_zeros_initializer(self):
        initializer = tf.initializers.Zeros(dtype=tf.float32)
        self.assertEqual(initializer((2, 3)), np.zeros((2, 3), 'float32'))
        self.assertEqual(initializer((2, 3)).dtype, 'float32')
        self.assertEqual(initializer((2, 3), tf.float64).dtype, 'float64')


class TestLinalgOps(OpTestCase):
    """Test linalg ops."""

    def test_eye(self):
        entries = [(2,), (2, 2), (2, 3), (3, 2)]
        for shape in entries:
            x = tf.eye(*shape, dtype=tf.float32)
            self.assertEqual(x, np.eye(*shape, dtype='float32'))


class TestMathOps(OpTestCase):
    """Test math ops."""

    # Testing shapes for binary ops.
    binary_test_shapes = [((2,), (2,)),
                          ((2, 3), (3,)),
                          ((2, 1), (2, 3)),
                          ((3,), (2, 3)),
                          ((2, 3), (2, 1)),
                          ((2, 1), (1, 3))]

    def test_abs(self):
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.abs(x), np.abs(data))

    def test_add(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1), new_tensor(data2)
            self.assertEqual(tf.math.add(a, b), data1 + data2)

    def test_add_n(self):
        for shape, _ in self.binary_test_shapes:
            data = arange(shape)
            x = new_tensor(data)
            self.assertEqual(tf.math.add_n([x, x, x]), data + data + data)

    def test_argmax(self):
        entries = [0, 1]
        for axis in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = tf.math.argmax(x, axis)
            self.assertEqual(y, np.argmax(data, axis))

    def test_argmin(self):
        entries = [0, 1]
        for axis in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = tf.math.argmin(x, axis)
            self.assertEqual(y, np.argmin(data, axis))

    def test_atan2(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.math.atan2(a, b)
            self.assertEqual(y, np.arctan2(data1, data2))

    def test_cast(self, test_float64=True):
        entries = [('float16', 'float32'), ('float32', 'float16'),
                   ('float32', 'float32'), ('float32', 'float64')]
        for in_type, out_type in entries:
            if not test_float64:
                if 'float64' in in_type or 'float64' in out_type:
                    continue
            data = np.array([-2., -1., 0., 1., 2.], dtype=in_type)
            x = new_tensor(data)
            y = tf.cast(x, tf.dtypes.DType(out_type))
            self.assertEqual(y, data.astype(out_type))

    def test_ceil(self):
        data = np.array([1.4, 1.7, 2.0], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.ceil(x), np.ceil(data))

    def test_cos(self):
        data = np.array([0., math.pi * 0.5, math.pi], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.cos(x), np.cos(data))

    def test_cumsum(self):
        entries = [(0, False, False),
                   (0, True, False),
                   (0, False, True),
                   (0, True, True)]
        for axis, exclusive, reverse in entries:
            data = arange((6,), 1)
            x = new_tensor(data)
            y = tf.math.cumsum(x, axis, exclusive, reverse)
            if reverse:
                data = np.flipud(data)
            if exclusive:
                data = np.array([0] + data[:-1].tolist(), 'float32')
            result = np.cumsum(data, axis)
            result = np.flipud(result) if reverse else result
            self.assertEqual(y, result)

    def test_divide(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.math.divide(a, b)
            self.assertEqual(y, data1 / data2)

    def test_equal(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.math.equal(a, b)
            self.assertEqual(y, np.equal(data1, data2))

    def test_exp(self):
        data = np.array([0., 1., 2.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.exp(x), np.exp(data))

    def test_floor(self):
        data = np.array([0.9, 1.4, 1.9], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.floor(x), np.floor(data))

    def test_greater(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.math.greater(a, b)
            self.assertEqual(y, np.greater(data1, data2))

    def test_greater_equal(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.math.greater_equal(a, b)
            self.assertEqual(y, np.greater_equal(data1, data2))

    def test_is_finite(self):
        data = np.array([0., float('nan'), float('inf')], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.is_finite(x), np.isfinite(data))

    def test_is_inf(self):
        data = np.array([0., 1., float('inf')], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.is_inf(x), np.isinf(data))

    def test_is_nan(self):
        data = np.array([0., 1., float('nan')], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.is_nan(x), np.isnan(data))

    def test_less(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.math.less(a, b)
            self.assertEqual(y, np.less(data1, data2))

    def test_less_equal(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.math.less_equal(a, b)
            self.assertEqual(y, np.less_equal(data1, data2))

    def test_linspace(self):
        entries = [([[0., 5.], [10., 40.], 5], {'axis': 0, 'dtype': 'float32'}),
                   ([[0., 5.], [10., 40.], 5], {'axis': 1, 'dtype': 'float32'}),
                   ([[0., 5.], [10., 40.], 5], {'axis': -1, 'dtype': 'float32'}),
                   ([[0.], [10.], 5], {'axis': 0, 'dtype': 'float32'}),
                   ([[0.], [10.], 5], {'axis': -1, 'dtype': 'float32'}),
                   ([0., 10., 5], {'axis': 0, 'dtype': 'float32'}),
                   ([0., 10., 5], {'axis': 0, 'dtype': 'int64'})]
        for (args, kwargs) in entries:
            data = np.linspace(*args, **kwargs)
            kwargs['dtype'] = tf.dtypes.DType(kwargs.pop('dtype'))
            x = tf.linspace(*args, **kwargs)
            self.assertEqual(x, data)

    def test_log(self):
        data = np.array([1., 2., 3.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.log(x), np.log(data))

    def test_matmul(self):
        entries = [((2, 3), (3, 4)),
                   ((1, 2, 3), (2, 3, 4)),
                   ((2, 2, 3), (1, 3, 4)),
                   ((2, 2, 3), (2, 3, 4)),
                   ((2, 1, 2, 3), (2, 3, 4)),
                   ((1, 2, 3), (2, 2, 3, 4)),
                   ((2, 1, 2, 3), (1, 2, 3, 4))]
        for a_shape, b_shape in entries:
            data1, data2 = arange(a_shape), arange(b_shape)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.linalg.matmul(a, b)
            self.assertEqual(y, np.matmul(data1, data2))
        entries = [((2,), (2,), (2, 1), (2, 1), (1, 1)),
                   ((2,), (2, 3), (2, 1), (2, 3), (1, 3)),
                   ((2, 3), (3,), (2, 3), (1, 3), (2, 1)),
                   ((2,), (4, 2, 3), (1, 2, 1), (4, 2, 3), (4, 1, 3)),
                   ((4, 2, 3), (3,), (4, 2, 3), (1, 1, 3), (4, 2, 1))]
        for a_shape, b_shape, da_shape, db_shape, dy_shape in entries:
            data1, data2 = arange(a_shape), arange(b_shape)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.linalg.matmul(a, b)
            self.assertEqual(y, np.matmul(data1, data2))

    def test_mul(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 10)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.math.multiply(a, b)
            self.assertEqual(y, data1 * data2)

    def test_negative(self):
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        y = tf.math.negative(x)
        self.assertEqual(y, np.negative(data))

    def test_not_equal(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1), new_tensor(data2)
            y = tf.math.not_equal(a, b)
            self.assertEqual(y, np.not_equal(data1, data2))

    def test_pow(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape, 1), arange(b_shape)
            a, b = new_tensor(data1), new_tensor(data2)
            self.assertEqual(tf.math.pow(a, b), np.power(data1, data2))

    def test_range(self):
        entries = [([5], {'dtype': 'int64'}),
                   ([0, 5], {'dtype': 'int64'}),
                   ([0, 5, 2], {'dtype': 'int64'}),
                   ([0., 1., 0.2], {'dtype': 'float32'})]
        for (args, kwargs) in entries:
            data = np.arange(*args, **kwargs)
            kwargs['dtype'] = tf.dtypes.DType(kwargs.pop('dtype'))
            self.assertEqual(tf.range(*args, **kwargs), data)

    def test_reciprocal(self):
        data = np.array([1., 2., 3.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.reciprocal(x), np.reciprocal(data))

    def test_reduce_max(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((1, 2), True), ((1, 2), False)]
        for axis, keepdims in entries:
            data = arange((2, 3, 3))
            x = new_tensor(data)
            y = tf.math.reduce_max(x, axis, keepdims=keepdims)
            self.assertEqual(y, np.max(data, axis, keepdims=keepdims))

    def test_reduce_mean(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((1, 2), True), ((1, 2), False)]
        for axis, keepdims in entries:
            data = arange((2, 3, 3))
            x = new_tensor(data)
            y = tf.math.reduce_mean(x, axis, keepdims=keepdims)
            self.assertEqual(y, np.mean(data, axis, keepdims=keepdims))

    def test_reduce_min(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((1, 2), True), ((1, 2), False)]
        for axis, keepdims in entries:
            data = arange((2, 3, 3))
            x = new_tensor(data)
            y = tf.math.reduce_min(x, axis, keepdims=keepdims)
            self.assertEqual(y, np.min(data, axis, keepdims=keepdims))

    def test_reduce_sum(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((1, 2), True), ((1, 2), False)]
        for axis, keepdims in entries:
            data = arange((2, 3, 3))
            x = new_tensor(data)
            y = tf.math.reduce_sum(x, axis, keepdims=keepdims)
            self.assertEqual(y, np.sum(data, axis, keepdims=keepdims))

    def test_reduce_variance(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((1, 2), True), ((1, 2), False)]
        for axis, keepdims in entries:
            data = arange((2, 3, 3))
            x = new_tensor(data)
            y = tf.math.reduce_variance(x, axis, keepdims=keepdims)
            self.assertEqual(y, np.var(data, axis, keepdims=keepdims))

    def test_round(self):
        data = np.array([0.9, 1.4, 1.9], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.round(x), np.round(data))

    def test_rsqrt(self):
        data = np.array([4., 9., 16], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.rsqrt(x), 1. / np.sqrt(data))

    def test_sigmoid(self):
        data = np.array([0.2, 0.4, 0.6, 0.8, 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.sigmoid(x), 1. / (1. + np.exp(-data)))

    def test_sign(self):
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.sign(x), np.sign(data))

    def test_sin(self):
        data = np.array([0., math.pi * 0.5, math.pi], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.sin(x), np.sin(data))

    def test_sqrt(self):
        data = np.array([4., 9., 16], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.sqrt(x), np.sqrt(data))

    def test_square(self):
        data = np.array([2., 3., 4.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.square(x), np.square(data))

    def test_sub(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape) + 1
            a, b = new_tensor(data1), new_tensor(data2)
            self.assertEqual(tf.math.subtract(a, b), data1 - data2)

    def test_tanh(self):
        data = np.array([0.2, 0.4, 0.6, 0.8, 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.math.tanh(x), np.tanh(data))


class TestNNOps(OpTestCase):
    """Test nn ops."""

    def test_avg_pool1d(self):
        entries = [((2, 2, 2), (2,), 2, 1, 'NCHW'),
                   ((2, 2, 2), (2,), 2, 1, 'NHWC')]
        for x_shape, kernel_shape, strides, pads, data_format in entries:
            data = arange(x_shape) * .1
            x = new_tensor(data)
            y = tf.nn.avg_pool1d(x, kernel_shape, strides,
                                 padding=pads, data_format=data_format)
            self.assertEqual(y, data / np.prod(kernel_shape))

    def test_avg_pool2d(self):
        entries = [((2, 2, 2, 2), (2, 2), 2, 1, 'NCHW'),
                   ((2, 2, 2, 2), (2, 2), 2, 1, 'NHWC')]
        for x_shape, kernel_shape, strides, pads, data_format in entries:
            data = arange(x_shape) * .1
            x = new_tensor(data)
            y = tf.nn.avg_pool2d(x, kernel_shape, strides,
                                 padding=pads, data_format=data_format)
            self.assertEqual(y, data / np.prod(kernel_shape))

    def test_avg_pool3d(self):
        entries = [((2, 2, 2, 2, 2), (2, 2, 2), 2, 1, 'NCHW'),
                   ((2, 2, 2, 2, 2), (2, 2, 2), 2, 1, 'NHWC')]
        for x_shape, kernel_shape, strides, pads, data_format in entries:
            data = arange(x_shape) * .1
            x = new_tensor(data)
            y = tf.nn.avg_pool3d(x, kernel_shape, strides,
                                 padding=pads, data_format=data_format)
            self.assertEqual(y, data / np.prod(kernel_shape))

    def test_bias_add(self):
        entries = [((2, 3), (3,), 'NCHW'),
                   ((2, 3, 4), (1, 3, 1), 'NCHW'),
                   ((2, 4, 3), (1, 1, 3), 'NHWC')]
        for x_shape, b_shape, data_format in entries:
            data1, data2 = arange(x_shape), arange(b_shape)
            x, b = new_tensor(data1), new_tensor(data2.flatten())
            y = tf.nn.bias_add(x, b, data_format=data_format)
            self.assertEqual(y, data1 + data2)

    def test_conv1d(self):
        entries = [((2, 2, 2), (3, 2, 1), 1, 1, 0, 1, 'NCHW'),
                   ((2, 2, 2), (3, 2, 3), 3, 1, 1, 1, 'NCHW'),
                   ((2, 2, 2), (3, 2, 1), 1, 1, 0, 1, 'NHWC'),
                   ((2, 2, 2), (3, 2, 3), 3, 1, (1, 1), 1, 'NHWC')]
        for (x_shape, w_shape, kernel_shape,
                strides, pads, dilations, data_format) in entries:
            data1, data2, = arange(x_shape) * .1, arange(w_shape) * .1
            x, w = new_tensor(data1), new_tensor(data2)
            _ = tf.nn.conv1d(x, w, strides=strides, padding=pads,
                             dilations=dilations, data_format=data_format)

    def test_conv1d_transpose(self):
        entries = [((2, 2, 2), (2, 3, 1), 1, 1, 0, 1, 'NCHW'),
                   ((2, 2, 2), (2, 3, 3), 3, 1, 1, 1, 'NCHW'),
                   ((2, 2, 2), (2, 3, 1), 1, 1, 0, 1, 'NHWC'),
                   ((2, 2, 2), (2, 3, 3), 3, 1, 1, 1, 'NHWC')]
        for (x_shape, w_shape, kernel_shape,
                strides, pads, dilations, data_format) in entries:
            data1, data2, = arange(x_shape) * .1, arange(w_shape) * .1
            x, w = new_tensor(data1), new_tensor(data2)
            _ = tf.nn.conv1d_transpose(x, w, strides=strides, padding=pads,
                                       dilations=dilations, data_format=data_format)

    def test_conv2d(self):
        entries = [((2, 2, 2, 2), (3, 2, 1, 1), 1, 1, 0, 1, 'NCHW'),
                   ((2, 2, 2, 2), (3, 2, 3, 3), 3, 1, (1, 1), 1, 'NCHW'),
                   ((2, 2, 2, 2), (3, 2, 1, 1), 1, 1, 0, 1, 'NHWC'),
                   ((2, 2, 2, 2), (3, 2, 3, 3), 3, None, 'SAME', 1, 'NHWC')]
        for (x_shape, w_shape, kernel_shape,
                strides, pads, dilations, data_format) in entries:
            data1, data2, = arange(x_shape) * .1, arange(w_shape) * .1
            x, w = new_tensor(data1), new_tensor(data2)
            _ = tf.nn.conv2d(x, w, strides=strides, padding=pads,
                             dilations=dilations, data_format=data_format)

    def test_conv2d_transpose(self):
        entries = [((2, 2, 2, 2), (2, 3, 1, 1), 1, 1, 0, 1, 'NCHW'),
                   ((2, 2, 2, 2), (2, 3, 3, 3), 3, 1, 1, 1, 'NCHW'),
                   ((2, 2, 2, 2), (2, 3, 1, 1), 1, 1, 0, 1, 'NHWC'),
                   ((2, 2, 2, 2), (2, 3, 3, 3), 3, 1, 1, 1, 'NHWC')]
        for (x_shape, w_shape, kernel_shape,
                strides, pads, dilations, data_format) in entries:
            data1, data2, = arange(x_shape) * .1, arange(w_shape) * .1
            x, w = new_tensor(data1), new_tensor(data2)
            _ = tf.nn.conv2d_transpose(x, w, strides=strides, padding=pads,
                                       dilations=dilations, data_format=data_format,
                                       output_shape=(2, 2))

    def test_conv3d(self):
        entries = [((2, 2, 2, 2, 2), (3, 2, 1, 1, 1), 1, 1, 0, 1, 'NCHW'),
                   ((2, 2, 2, 2, 2), (3, 2, 3, 3, 3), 3, 1, 1, 1, 'NCHW'),
                   ((2, 2, 2, 2, 2), (3, 2, 1, 1, 1), 1, 1, 0, 1, 'NHWC'),
                   ((2, 2, 2, 2, 2), (3, 2, 3, 3, 3), 3, 1, 1, 1, 'NHWC')]
        for (x_shape, w_shape, kernel_shape,
                strides, pads, dilations, data_format) in entries:
            data1, data2, = arange(x_shape) * .1, arange(w_shape) * .1
            x, w = new_tensor(data1), new_tensor(data2)
            _ = tf.nn.conv3d(x, w, strides=strides, padding=pads,
                             dilations=dilations, data_format=data_format)

    def test_conv3d_transpose(self):
        entries = [((2, 2, 2, 2, 2), (2, 3, 1, 1, 1), 1, 1, 0, 1, 'NCHW'),
                   ((2, 2, 2, 2, 2), (2, 3, 3, 3, 3), 3, 1, 1, 1, 'NCHW'),
                   ((2, 2, 2, 2, 2), (2, 3, 1, 1, 1), 1, 1, 0, 1, 'NHWC'),
                   ((2, 2, 2, 2, 2), (2, 3, 3, 3, 3), 3, 1, 1, 1, 'NHWC')]
        for (x_shape, w_shape, kernel_shape,
                strides, pads, dilations, data_format) in entries:
            data1, data2, = arange(x_shape) * .1, arange(w_shape) * .1
            x, w = new_tensor(data1), new_tensor(data2)
            _ = tf.nn.conv3d_transpose(x, w, strides=strides, padding=pads,
                                       dilations=dilations, data_format=data_format)

    def test_dropout(self):
        data = uniform((2, 3))
        x = new_tensor(data)
        self.assertEqual(tf.nn.dropout(x, 0.0), data)

    def test_elu(self):
        alpha = 1.
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        y = tf.nn.elu(x, alpha)
        result = np.maximum(data, 0.) + alpha * (np.exp(np.minimum(data, 0.)) - 1.)
        self.assertEqual(y, result)

    def test_fused_batch_norm(self):
        eps = 1e-5
        entries = [((4, 3), (3,), -1, 0),
                   ((4, 3), (3,), -1, 1),
                   ((4, 3, 2), (1, 3, 1), 1, 0),
                   ((4, 3, 2), (1, 3, 1), 1, 1),
                   ((4, 2, 3), (1, 1, 3), -1, 0),
                   ((4, 2, 3), (1, 1, 3), -1, 1)]
        for x_shape, w_shape, axis, use_stats in entries:
            data1 = arange(x_shape) * .1
            data2, data3 = arange(w_shape, 1) * .1, arange(w_shape) * .1
            data4, data5 = arange(w_shape) * .1, arange(w_shape, 1) * .1
            x = new_tensor(data1)
            w, b = new_tensor(data2.flatten()), new_tensor(data3.flatten())
            rm, rv = new_tensor(data4.flatten()), new_tensor(data5.flatten())
            y = tf.nn.fused_batch_norm(
                x, w, b, rm, rv, data_format='NCHW' if axis == 1 else 'NHWC',
                is_training=use_stats == 0, epsilon=eps)
            if use_stats == 0:
                axes = list(range(0, len(data1.shape)))
                axes.pop(axis)
                mean = broadcast_like(np.mean(data1, tuple(axes)), data1, axes)
                sig = broadcast_like(np.sqrt(np.var(data1, tuple(axes)) + eps), data1, axes)
                result = (data1 - mean) / sig
            else:
                sig = np.sqrt(data5 + eps)
                result = (data1 - data4) / sig
            result = result * data2 + data3
            self.assertEqual(y, result)

    def test_gelu(self):
        data = np.array([-1., 0., 1.], 'float32')
        cdf = data.copy()
        for i in range(data.size):
            cdf[i] = 0.5 * (1 + math.erf(data[i] * 0.7071067811865475))
        for approximate in (False, True):
            x = new_tensor(data)
            y = tf.nn.gelu(x, approximate=approximate)
            self.assertEqual(y, data * cdf, prec=0.001 if approximate else None)

    def test_leaky_relu(self):
        alpha = 0.2
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        y = tf.nn.leaky_relu(x, alpha)
        result = np.maximum(data, 0.) + np.minimum(data, 0.) * alpha
        self.assertEqual(y, result)

    def test_log_softmax(self):
        data = np.log(np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32'))
        x = new_tensor(data)
        self.assertEqual(tf.nn.log_softmax(x), data)

    def test_l2_loss(self):
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.nn.l2_loss(x), np.square(data).sum() * 0.5)

    def test_l2_normalize(self):
        data = arange((6,))
        x = new_tensor(data)
        y = tf.nn.l2_normalize(x, axis=0, epsilon=1e-12)
        norm = np.sqrt(np.square(data).sum())
        self.assertEqual(y, data / max(norm, 1e-12))

    def test_max_pool1d(self):
        entries = [((2, 2, 2), (2,), 2, 1, 'NCHW'),
                   ((2, 2, 2), (2,), 2, 1, 'NHWC')]
        for x_shape, kernel_shape, strides, pads, data_format in entries:
            data = arange(x_shape) * .1
            x = new_tensor(data)
            y = tf.nn.max_pool1d(x, kernel_shape, strides,
                                 padding=pads, data_format=data_format)
            self.assertEqual(y, data)

    def test_max_pool2d(self):
        entries = [((2, 2, 2, 2), (2, 2), 2, 1, 'NCHW'),
                   ((2, 2, 2, 2), (2, 2), 2, 1, 'NHWC')]
        for x_shape, kernel_shape, strides, pads, data_format in entries:
            data = arange(x_shape) * .1
            x = new_tensor(data)
            y = tf.nn.max_pool2d(x, kernel_shape, strides,
                                 padding=pads, data_format=data_format)
            self.assertEqual(y, data)

    def test_max_pool3d(self):
        entries = [((2, 2, 2, 2, 2), (2, 2, 2), 2, 1, 'NCHW'),
                   ((2, 2, 2, 2, 2), (2, 2, 2), 2, 1, 'NHWC')]
        for x_shape, kernel_shape, strides, pads, data_format in entries:
            data = arange(x_shape) * .1
            x = new_tensor(data)
            y = tf.nn.max_pool3d(x, kernel_shape, strides,
                                 padding=pads, data_format=data_format)
            self.assertEqual(y, data)

    def test_relu(self):
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.nn.relu(x), np.maximum(data, 0.))

    def test_relu6(self):
        data = np.array([-1., 0., 1., 6., 7.], 'float32')
        x = new_tensor(data)
        self.assertEqual(tf.nn.relu6(x), np.minimum(np.maximum(data, 0.), 6.))

    def test_selu(self):
        alpha, gamma = 1.67326, 1.0507
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        y = tf.nn.selu(x)
        result = gamma * (
            np.maximum(data, 0.) +
            alpha * (np.exp(np.minimum(data, 0.)) - 1.))
        self.assertEqual(y, result)

    def test_silu(self):
        data = np.array([-3., -2., -1., 0., 1., 2., 3.], 'float32')
        x = new_tensor(data)
        y = tf.nn.silu(x)
        result = data * (1. / (1. + np.exp(-data)))
        self.assertEqual(y, result)

    def test_sigmoid_cross_entropy_with_logits(self):
        data1 = np.array([[0.2], [0.5], [0.7]], 'float32')
        data2 = -np.log(1. / data1 - 1.)
        data3 = np.array([[0], [1], [0]], 'float32')
        x, y = new_tensor(data2), new_tensor(data3)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(y, x)
        result = -(data3 * np.log(data1) + (1 - data3) * np.log(1 - data1))
        self.assertEqual(loss, result)

    def test_softmax(self):
        data = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32')
        x = new_tensor(np.log(data))
        self.assertEqual(tf.nn.softmax(x), data)

    def test_softmax_cross_entropy_with_logits(self):
        data1 = np.log(np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32'))
        data2 = np.array([0, 1], 'int64')
        data3 = np.eye(3, dtype='float32')[data2]
        result = -data1[np.arange(2), data2]
        x, y = new_tensor(data1), new_tensor(data3)
        loss = tf.nn.softmax_cross_entropy_with_logits(y, x)
        self.assertEqual(loss, result)

    def test_sparse_softmax_cross_entropy_with_logits(self):
        data1 = np.log(np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32'))
        data2 = np.array([0, 1], 'int64')
        result = -data1[np.arange(2), data2]
        x, y = new_tensor(data1), new_tensor(data2)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y, x)
        self.assertEqual(loss, result)

    def test_top_k(self):
        entries = [(2, -1, True)]
        for k, axis, largest in entries:
            data = uniform((5, 10))
            x = new_tensor(data)
            y = tf.math.top_k(x, k)
            axis = axis if axis is not None else -1
            result1 = np.argsort(-data if largest else data, axis=axis)
            result2 = np.take(result1, np.arange(k), axis=axis)
            self.assertEqual(y[1], result2)


class TestRandomOps(OpTestCase):
    """Test random ops."""

    def test_random_normal(self):
        x = tf.random.normal((2, 3), dtype=tf.float64)
        self.assertEqual(x.dtype, 'float64')

    def test_random_uniform(self):
        x = tf.random.uniform((2, 3), dtype=tf.float64)
        self.assertEqual(x.dtype, 'float64')

    def test_truncated_normal(self):
        x = tf.random.truncated_normal((2, 3), dtype=tf.float64)
        self.assertEqual(x.dtype, 'float64')


class TestSortOps(OpTestCase):
    """Test sort ops."""

    def test_argsort(self):
        entries = [(None, True),
                   (0, True),
                   (-1, True),
                   (0, False),
                   (-1, False)]
        for axis, descending in entries:
            direction = 'DESCENDING' if descending else 'ASCENDING'
            data = uniform((5, 10))
            x = new_tensor(data)
            val = tf.sort(x, axis=axis, direction=direction)
            idx = tf.argsort(x, axis=axis, direction=direction)
            axis = axis if axis is not None else -1
            result_val = np.sort(-data if descending else data, axis=axis)
            result_val = -result_val if descending else result_val
            result_idx = np.argsort(-data if descending else data, axis=axis)
            result_idx = np.take(result_idx, np.arange(data.shape[axis]), axis=axis)
            self.assertEqual(val, result_val)
            self.assertEqual(idx, result_idx)


class TestResources(OpTestCase):
    """Test resources."""

    def test_variable(self):
        self.assertEqual(tf.Variable(np.zeros((2,))).trainable, True)
        self.assertEqual(tf.Variable(tf.zeros((2, 1)), shape=(2,)).shape, (2,))
        self.assertEqual(tf.Variable(1, shape=(2,)).shape, (2,))
        repr(tf.Variable([2, 3, 3]))
        self.assertEqual(repr(tf.Variable(1)), '1')


def arange(shape, start=0, dtype='float32'):
    """Return the arange data with given shape."""
    return np.arange(start, start + int(np.prod(shape)), dtype=dtype).reshape(shape)


def broadcast_like(data, other, axes):
    """Broadcast data like the other."""
    shape = list(other.shape[:])
    for i in nest.flatten(axes):
        shape[i] = 1
    return data.reshape(shape) * np.ones_like(other, data.dtype)


def new_tensor(data):
    """Create a new tensor for current execution."""
    return tf.constant(data)


def process_index(item):
    """Process and normalize the index."""
    if not isinstance(item, (slice, tuple)):
        if not isinstance(item, int):
            raise ValueError('The index should be a integer.')
        item = (item,)
    if not isinstance(item, tuple):
        item = tuple([item])
    starts, sizes = [], []
    for i, ele in enumerate(item):
        if isinstance(ele, slice):
            if ele.start is None:
                starts.append(0)
            else:
                starts.append(ele.start)
            if ele.stop is None:
                sizes.append(-1)
            else:
                sizes.append(ele.stop - starts[-1])
                if sizes[-1] == 0:
                    raise ValueError(
                        'The starts and ends of axis {} can not be equal, got {}:{}.'
                        .format(i, starts[-1], ele.stop))
            if ele.step is not None:
                raise NotImplementedError
        elif isinstance(ele, int):
            starts.append(ele)
            sizes.append(0)
        else:
            raise TypeError('Unsupported type of index: {}'.format(type(ele)))
    return starts, sizes


def uniform(shape, dtype='float32'):
    """Return the uniform data with given shape."""
    return np.random.uniform(-1., 1., size=shape).astype(dtype)


if __name__ == '__main__':
    run_tests()
