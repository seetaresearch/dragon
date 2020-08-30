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
"""Test the ops module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import unittest

import numpy as np

from dragon.core.util import nest
from dragon.core.testing.unittest.common_utils import run_tests
from dragon.vm import torch

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
        for i, input in enumerate(inputs):
            if isinstance(input, torch.Tensor):
                inputs[i] = input.numpy()
        first = inputs[:num_first] if num_first > 1 else inputs[0]
        second = inputs[num_first:len(inputs)] if num_second > 1 else inputs[num_first]
        if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            super(OpTestCase, self).assertEqual(first.shape, second.shape)
            if first.dtype == np.bool and second.dtype == np.bool:
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


class TestTensorOps(OpTestCase):
    """Test the tensor ops."""

    # Testing shapes for binary ops
    unary_test_shapes = [(2,)]

    # Testing shapes for binary ops
    binary_test_shapes = [((2,), (2,)), ((2, 3), (3,)), ((2, 3), (2, 1))]

    def test_abs(self):
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.abs(), np.abs(data))

    def test_add(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a + b, data1 + data2)
            a += b
            self.assertEqual(a, data1 + data2)

    def test_argmax(self):
        entries = [(0, True), (0, False), (1, True), (1, False)]
        for axis, keep_dims in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            result = np.argmax(data, axis)
            if keep_dims:
                result = np.expand_dims(result, axis)
            self.assertEqual(x.argmax(axis, keep_dims), result)

    def test_argmin(self):
        entries = [(0, True), (0, False), (1, True), (1, False)]
        for axis, keep_dims in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            result = np.argmin(data, axis)
            if keep_dims:
                result = np.expand_dims(result, axis)
            self.assertEqual(x.argmin(axis, keep_dims), result)

    def test_bitwise_not(self):
        for shape in self.unary_test_shapes:
            data = np.random.binomial(1, 0.5, shape).astype('bool')
            x = new_tensor(data)
            self.assertEqual(x.bitwise_not(), np.invert(data))
            x.bitwise_not_()
            self.assertEqual(x, np.invert(data))

    def test_bitwise_xor(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1 = np.random.binomial(1, 0.5, a_shape).astype('bool')
            data2 = np.random.binomial(1, 0.5, b_shape).astype('bool')
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.bitwise_xor(b), np.bitwise_xor(data1, data2))
            a.bitwise_xor_(b)
            self.assertEqual(a, np.bitwise_xor(data1, data2))

    def test_ceil(self):
        data = np.array([1.4, 1.7, 2.0])
        x = new_tensor(data)
        self.assertEqual(x.ceil(), np.ceil(data))
        x.ceil_()
        self.assertEqual(x, np.ceil(data))

    def test_chunk(self):
        data = arange((2, 3))
        x = new_tensor(data)
        y = x.chunk(2, 1)
        self.assertEqual(y, [np.split(data, (2,), axis=1)])

    def test_clamp(self):
        entries = [(None, None), (2, None), (None, 4), (2, 4)]
        for low, high in entries:
            data = arange((6,))
            x = new_tensor(data)
            result = np.clip(data, low, high) if low or high else data
            self.assertEqual(x.clamp(low, high), result)
            x.clamp_(low, high)
            self.assertEqual(x, result)

    def test_copy(self):
        data1, data2 = arange((2,)), arange((2, 3))
        a, b = new_tensor(data1, False), new_tensor(data2, False)
        self.assertEqual(a.copy_(b), data2)

    def test_cos(self):
        data = np.array([0., math.pi * 0.5, math.pi], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.cos(), np.cos(data))

    def test_cum_sum(self):
        data = arange((6,), 1)
        x = new_tensor(data)
        self.assertEqual(x.cumsum(0), np.cumsum(data, 0))

    def test_div(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a / b, data1 / data2)
            a /= b
            self.assertEqual(a, data1 / data2)

    def test_equal(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1 = uniform(a_shape)
            data2 = dropout(data1, drop_ratio=0.5)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.eq(b), np.equal(data1, data2))

    def test_exp(self):
        data = np.array([0., 1., 2.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.exp(), np.exp(data))

    def test_expand(self):
        entries = [(2, 2, 3, 1),
                   (1, 2, 3, 2),
                   (2, 2, 3, 2),
                   (2, 1, 2, 3, 1)]
        for shape in entries:
            data = np.arange(6).astype('float32').reshape((1, 2, 3, 1))
            x = new_tensor(data)
            self.assertEqual(x.expand(shape), np.broadcast_to(data, shape))
            self.assertEqual(x.expand_as(x.expand(shape)), np.broadcast_to(data, shape))

    def test_fill(self):
        entries = [((2, 3), 1), ((2, 3), 1.)]
        for shape, value in entries:
            data = np.zeros(shape)
            x = new_tensor(data)
            x.fill_(value)
            data.fill(value)
            self.assertEqual(x, data)

    def test_flatten(self):
        data = arange((1, 2, 3))
        x = new_tensor(data)
        self.assertEqual(x.flatten(), data.flatten())
        x.flatten_(-3, -2)
        self.assertEqual(x, data.reshape((2, 3)))

    def test_floor(self):
        data = np.array([0.9, 1.4, 1.9])
        x = new_tensor(data)
        self.assertEqual(x.floor(), np.floor(data))
        x.floor_()
        self.assertEqual(x, np.floor(data))

    def test_getitem(self):
        data1, data2 = arange((2, 3)), arange((2,), dtype='int64')
        x, index = new_tensor(data1), new_tensor(data2)
        self.assertEqual(x[x > 2], data1[data1 > 2])
        entries = [0,
                   slice(None, None, None),
                   slice(0, None, None),
                   slice(0, 0, None),
                   slice(0, 1, None),
                   slice(0, 1, 1)]
        for item in entries:
            try:
                self.assertEqual(x.__getitem__(item), data1.__getitem__(item))
            except (NotImplementedError, ValueError):
                pass
        self.assertEqual(x[index], data1[data2])
        self.assertEqual(x[:, index], data1[:, data2])
        entries = [x,
                   (slice(1, None, None), index),
                   (1, index),
                   (index, index)]
        for item in entries:
            try:
                x.__getitem__(item)
            except TypeError:
                pass

    def test_greater(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = uniform(a_shape), uniform(b_shape)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a > b, np.greater(data1, data2))

    def test_greater_equal(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = uniform(a_shape), uniform(b_shape)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a >= b, np.greater_equal(data1, data2))

    def test_index_select(self):
        entries = [1, (1, 2)]
        for axis in entries:
            data = arange((1, 2, 3, 4))
            index = np.array([0, 1, 1], dtype='int64')
            axes = nest.flatten(axis)
            if len(axes) > 1:
                flatten_shape = \
                    data.shape[:axes[0]] + \
                    (int(np.prod(data.shape[axes[0]:axes[-1] + 1])),) + \
                    data.shape[axes[-1] + 1:]
            else:
                flatten_shape = data.shape[:]
            for i in index:
                slices = [slice(None, None, None)] * (len(flatten_shape) - 1)
                slices.insert(axes[0], i)
            x = new_tensor(data)
            x_index = new_tensor(index, False)
            y = x.index_select(axis, x_index)
            self.assertEqual(
                y, np.take(data.reshape(flatten_shape), index, axis=axes[0]))

    def test_less(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = uniform(a_shape), uniform(b_shape)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a < b, np.less(data1, data2))

    def test_less_equal(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = uniform(a_shape), uniform(b_shape)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a <= b, np.less_equal(data1, data2))

    def test_log(self):
        data = np.array([1., 2., 3.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.log(), np.log(data))

    def test_log_sum_exp(self):
        data = np.array([1., 2., 3.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.logsumexp(0), np.log(np.sum(np.exp(data))))

    def test_masked_fill(self):
        data = arange((2, 3))
        x = new_tensor(data)
        x.masked_fill_(x > 2, 0)
        data[data > 2] = 0
        self.assertEqual(x, data)

    def test_max(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for axis, keep_dims in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = x.max(axis, keepdim=keep_dims)
            result = np.max(data, axis, keepdims=keep_dims)
            self.assertEqual(y, result)

    def test_mean(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for axis, keep_dims in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = x.mean(axis, keepdim=keep_dims)
            result = np.mean(data, axis, keepdims=keep_dims)
            self.assertEqual(y, result)

    def test_min(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for axis, keep_dims in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = x.min(axis, keepdim=keep_dims)
            result = np.min(data, axis, keepdims=keep_dims)
            self.assertEqual(y, result)

    def test_mul(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a * b, data1 * data2)
            a *= b
            self.assertEqual(a, data1 * data2)

    def test_multinomial(self):
        data = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        x = new_tensor(data)
        y = x.multinomial(2)
        self.assertEqual(y.shape, (2, 2))

    def test_narrow(self):
        data = arange((2, 3))
        x = new_tensor(data)
        self.assertEqual(x.narrow(0, 1, 1), data[1:2, :])

    def test_not_equal(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1 = uniform(a_shape)
            data2 = dropout(data1, drop_ratio=0.5)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.ne(b), np.not_equal(data1, data2))

    def test_neg(self):
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(-x, -data)

    def test_non_zero(self):
        data = arange((2, 3))
        x = new_tensor(data)
        self.assertEqual((x > 2).nonzero(), np.stack(np.nonzero(data > 2), axis=1))

    def test_normal(self):
        data = arange((2, 3))
        x = new_tensor(data)
        x.normal_()

    def test_permute(self):
        entries = [(0, 2, 1), None]
        for perm in entries:
            data = arange((2, 3, 4))
            x = new_tensor(data)
            if perm is None:
                self.assertEqual(x.permute(), np.transpose(data))
            else:
                self.assertEqual(x.permute(*perm), np.transpose(data, perm))

    def test_pow(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape, 1), arange(b_shape)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.pow(b), np.power(data1, data2))

    def test_reciprocal(self):
        data = np.array([1., 2., 3.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.reciprocal(), np.reciprocal(data))
        x.reciprocal_()
        self.assertEqual(x, np.reciprocal(data))

    def test_repeat(self):
        entries = [(2,), (1, 1), (1, 2), (2, 1), (2, 2)]
        for repeats in entries:
            data = arange((2, 2))
            x = new_tensor(data)
            y = x.repeat(repeats)
            repeats = repeats + (1,) * (len(data.shape) - len(repeats))
            self.assertEqual(y, np.tile(data, repeats))

    def test_reshape(self):
        entries = [(0, 0), (0, -1)]
        for shape in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = x.reshape(shape)
            self.assertEqual(y, data.reshape(y.shape))
            x.reshape_(shape)
            self.assertEqual(x, data.reshape(y.shape))
            self.assertEqual(x.view(data.shape), data)
            x.view_(data.shape)
            self.assertEqual(x, data)
            self.assertEqual(x.view_as(x), data)

    def test_round(self):
        data = np.array([0.9, 1.4, 1.9], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.round(), np.round(data))
        x.round_()
        self.assertEqual(x, np.round(data))

    def test_rsqrt(self):
        data = np.array([4., 9., 16], 'float32')
        x = new_tensor(data)
        result = 1. / np.sqrt(data)
        self.assertEqual(x.rsqrt(), result)
        x.rsqrt_()
        self.assertEqual(x, result)

    def test_setitem(self):
        data = arange((2, 3))
        x = new_tensor(data)
        x[x > 2] = 0
        data[data > 2] = 0
        self.assertEqual(x, data)
        entries = [0,
                   slice(None, None, None),
                   slice(0, None, None),
                   slice(0, 0, None),
                   slice(0, 1, None),
                   slice(0, 1, 1),
                   data,
                   (data, data)]
        for item in entries:
            try:
                x.__setitem__(item, 0)
                data.__setitem__(item, 0)
                self.assertEqual(x, data)
            except (NotImplementedError, ValueError, TypeError):
                pass

    def test_sign(self):
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.sign(), np.sign(data))
        x.sign_()
        self.assertEqual(x, np.sign(data))

    def test_sin(self):
        data = np.array([0., math.pi * 0.5, math.pi], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.sin(), np.sin(data))

    def test_sqrt(self):
        data = np.array([4., 9., 16], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.sqrt(), np.sqrt(data))
        x.sqrt_()
        self.assertEqual(x, np.sqrt(data))

    def test_squeeze(self):
        entries = [((2, 1, 3), 1), ((1, 2, 1, 3), (0, 2)), ((3, 1, 2, 1), (1,))]
        for shape, axis in entries:
            data = arange(shape)
            x = new_tensor(data)
            self.assertEqual(x.squeeze(axis), np.squeeze(data, axis))
            x.squeeze_(axis)
            self.assertEqual(x, np.squeeze(data, axis))

    def test_sub(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a - b, data1 - data2)
            a -= b
            self.assertEqual(a, data1 - data2)

    def test_topk(self):
        entries = [(2, None, True),
                   (2, 0, True),
                   (2, -1, True),
                   (2, 0, False),
                   (2, -1, False)]
        for k, axis, largest in entries:
            data = uniform((5, 10))
            x = new_tensor(data)
            y = x.topk(k, axis, largest)[1]
            axis = axis if axis is not None else -1
            result = np.argsort(-data if largest else data, axis=axis)
            result = np.take(result, np.arange(k), axis=axis)
            self.assertEqual(y, result)

    def test_type(self):
        entries = [('bool', 'bool'),
                   ('byte', 'uint8'),
                   ('char', 'int8'),
                   ('double', 'float64'),
                   ('float', 'float32'),
                   ('half', 'float16'),
                   ('int', 'int32'),
                   ('long', 'int64')]
        for name, dtype in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            self.assertEqual(getattr(x, name)(), data.astype(dtype))
            getattr(x, name + '_')()
            self.assertEqual(x, data.astype(dtype))
            x.type(dtype)
            self.assertEqual(x.type(), dtype)

    def test_uniform(self):
        data = arange((2, 3))
        x = new_tensor(data)
        x.uniform_()

    def test_unsqueeze(self):
        entries = [1, -1]
        for axis in entries:
            data = arange((2, 3, 4))
            x = new_tensor(data)
            self.assertEqual(x.unsqueeze(axis), np.expand_dims(data, axis=axis))
            x.unsqueeze_(axis)
            self.assertEqual(x, np.expand_dims(data, axis=axis))

    def test_where(self):
        entries = [((6,), (6,))]
        for a_shape, b_shape in entries:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            data3 = data1 > 1
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            c = new_tensor(data3, False)
            self.assertEqual(a.where(c, b), np.where(data3, data1, data2))


class TestTorchOps(OpTestCase):
    """Test the builtin torch ops."""

    def test_ones_like(self):
        data = np.ones((2, 3), dtype='float32')
        x = new_tensor(data)
        self.assertEqual(torch.ones_like(x), data)

    def test_rand(self):
        self.assertEqual(torch.rand(2, 3).shape, (2, 3))

    def test_randn(self):
        self.assertEqual(torch.randn(2, 3).shape, (2, 3))

    def test_randperm(self):
        self.assertEqual(torch.randperm(4).shape, (4,))

    def test_zeros_like(self):
        data = np.zeros((2, 3), dtype='float32')
        x = new_tensor(data)
        self.assertEqual(torch.zeros_like(x), data)


def arange(shape, start=0, dtype='float32'):
    """Return the arange data with given shape."""
    return np.arange(start, start + int(np.prod(shape)), dtype=dtype).reshape(shape)


def dropout(data, drop_ratio=0.5):
    """Return the random dropped data."""
    return data * np.random.binomial(1, 1. - drop_ratio, data.shape).astype(data.dtype)


def new_tensor(data, requires_grad=False):
    """Create a new tensor from data."""
    return torch.tensor(data, dtype=data.dtype, requires_grad=requires_grad)


def uniform(shape, dtype='float32'):
    """Return the uniform data with given shape."""
    return np.random.uniform(-1., 1., size=shape).astype(dtype)


if __name__ == '__main__':
    run_tests()
