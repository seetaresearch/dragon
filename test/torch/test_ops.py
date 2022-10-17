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

import itertools
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
        for i, x in enumerate(inputs):
            if isinstance(x, torch.Tensor):
                inputs[i] = x.numpy()
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


class TestTensorOps(OpTestCase):
    """Test tensor ops."""

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
            self.assertEqual(1 + a, 1 + data1)
            a += b
            self.assertEqual(a, data1 + data2)

    def test_addmm(self):
        entries = [((2, 3), (3, 4), (2, 4))]
        for a_shape, b_shape, c_shape in entries:
            data1, data2 = arange(a_shape), arange(b_shape)
            data3 = arange(c_shape)
            a, b = new_tensor(data1), new_tensor(data2)
            c = new_tensor(data3)
            y = c.addmm(a, b)
            self.assertEqual(y, np.matmul(data1, data2) + data3)

    def test_argmax(self):
        entries = [(0, True), (0, False), (1, True), (1, False)]
        for axis, keepdims in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            result = np.argmax(data, axis)
            if keepdims:
                result = np.expand_dims(result, axis)
            self.assertEqual(x.argmax(axis, keepdims), result)

    def test_argmin(self):
        entries = [(0, True), (0, False), (1, True), (1, False)]
        for axis, keepdims in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            result = np.argmin(data, axis)
            if keepdims:
                result = np.expand_dims(result, axis)
            self.assertEqual(x.argmin(axis, keepdims), result)

    def test_atan2(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.atan2(b), np.arctan2(data1, data2))

    def test_baddbmm(self):
        entries = [((2, 2, 3), (2, 3, 4), (2, 2, 4))]
        for a_shape, b_shape, c_shape in entries:
            data1, data2 = arange(a_shape), arange(b_shape)
            data3 = arange(c_shape)
            a, b = new_tensor(data1), new_tensor(data2)
            c = new_tensor(data3)
            y = c.baddbmm(a, b)
            self.assertEqual(y, np.matmul(data1, data2) + data3)
            c.baddbmm_(a, b)
            self.assertEqual(c, np.matmul(data1, data2) + data3)

    def test_bitwise_and(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1 = arange(a_shape, dtype='int32')
            data2 = arange(b_shape, 1, dtype='int32')
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a & b, np.bitwise_and(data1, data2))
            a &= b
            self.assertEqual(a, np.bitwise_and(data1, data2))

    def test_bitwise_not(self):
        for shape in self.unary_test_shapes:
            data = np.random.binomial(1, 0.5, shape).astype('bool')
            x = new_tensor(data)
            self.assertEqual(~x, np.invert(data))
            x.bitwise_not_()
            self.assertEqual(x, np.invert(data))

    def test_bitwise_or(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1 = arange(a_shape, dtype='int32')
            data2 = arange(b_shape, 1, dtype='int32')
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a | b, np.bitwise_or(data1, data2))
            a |= b
            self.assertEqual(a, np.bitwise_or(data1, data2))

    def test_bitwise_xor(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1 = arange(a_shape, dtype='int32')
            data2 = arange(b_shape, 1, dtype='int32')
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a ^ b, np.bitwise_xor(data1, data2))
            a ^= b
            self.assertEqual(a, np.bitwise_xor(data1, data2))

    def test_bmm(self):
        test_shapes = [((1, 2, 3), (2, 3, 4)),
                       ((2, 2, 3), (1, 3, 4)),
                       ((2, 2, 3), (2, 3, 4)),
                       ((2, 1, 2, 3), (2, 3, 4)),
                       ((1, 2, 3), (2, 2, 3, 4)),
                       ((2, 1, 2, 3), (1, 2, 3, 4))]
        for a_shape, b_shape in test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.bmm(b), np.matmul(data1, data2))

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
            self.assertEqual(a == b, np.equal(data1, data2))

    def test_exp(self):
        data = np.array([0., 1., 2.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.exp(), np.exp(data))
        x.exp_()
        self.assertEqual(x, np.exp(data))

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

    def test_eye(self):
        entries = [(2,), (2, 2), (2, 3), (3, 2)]
        for shape in entries:
            x = torch.eye(*shape, dtype='float32')
            self.assertEqual(x, np.eye(*shape, dtype='float32'))

    def test_fill(self):
        entries = [((2, 3), 1), ((2, 3), 1.)]
        for shape, value in entries:
            data = np.zeros(shape)
            x = new_tensor(data)
            x.fill_(value)
            data.fill(value)
            self.assertEqual(x, data)

    def test_full(self):
        entries = [((2, 3), 1), ((2, 3), 1.)]
        for shape, value in entries:
            data = np.zeros(shape)
            x = torch.full((1,), 0).new_full(shape, value)
            data.fill(value)
            self.assertEqual(x, data)
            self.assertEqual(torch.empty(1).new_ones(shape), np.ones(shape))
            self.assertEqual(torch.empty(1).new_zeros(shape), np.zeros(shape))
            self.assertEqual(torch.full_like(x, 0), np.zeros(shape))

    def test_flatten(self):
        data = arange((1, 2, 3))
        x = new_tensor(data)
        self.assertEqual(x.flatten(), data.flatten())
        x.flatten_(-3, -2)
        self.assertEqual(x, data.reshape((2, 3)))

    def test_flip(self):
        data = arange((2, 3, 4))
        x = new_tensor(data)
        self.assertEqual(x.flip((1, 2)), np.flip(data, (1, 2)))
        self.assertEqual(x.fliplr(), np.fliplr(data))
        self.assertEqual(x.flipud(), np.flipud(data))

    def test_floor(self):
        data = np.array([0.9, 1.4, 1.9])
        x = new_tensor(data)
        self.assertEqual(x.floor(), np.floor(data))
        x.floor_()
        self.assertEqual(x, np.floor(data))

    def test_gather(self):
        for axis in range(0, 1):
            data1 = arange((2, 4))
            data2 = np.array([[0, 1, 1, 0], [1, 1, 0, 0]], 'int64')
            x, index = new_tensor(data1), new_tensor(data2)
            y = x.gather(axis, index)
            result = np.zeros_like(data2)
            for i, j in itertools.product(*[range(d) for d in data2.shape]):
                if axis == 0:
                    result[i, j] = data1[data2[i, j], j]
                else:
                    result[i, j] = data1[i, data2[i, j]]
            self.assertEqual([y], [result])

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
            except (TypeError, NotImplementedError):
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
            index = np.array([0, 1, 1], 'int64')
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

    def test_isfinite(self):
        data = np.array([0., float('nan'), float('inf')])
        x = new_tensor(data)
        self.assertEqual(x.isfinite(), np.isfinite(data))

    def test_isinf(self):
        data = np.array([0., 1., float('inf')])
        x = new_tensor(data)
        self.assertEqual(x.isinf(), np.isinf(data))

    def test_isnan(self):
        data = np.array([0., 1., float('nan')])
        x = new_tensor(data)
        self.assertEqual(x.isnan(), np.isnan(data))

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
        x.log_()
        self.assertEqual(x, np.log(data))

    def test_logical_and(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.logical_and(b), np.logical_and(data1, data2))

    def test_logical_not(self):
        for shape in self.unary_test_shapes:
            data = arange(shape)
            x = new_tensor(data)
            self.assertEqual(x.logical_not(), np.logical_not(data))

    def test_logical_or(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.logical_or(b), np.logical_or(data1, data2))

    def test_logical_xor(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.logical_xor(b), np.logical_xor(data1, data2))

    def test_log_sum_exp(self):
        data = np.array([1., 2., 3.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.logsumexp(0), np.log(np.sum(np.exp(data))))

    def test_masked_fill(self):
        data = arange((2, 3))
        x = new_tensor(data)
        mask = x > 2
        y = x.masked_fill(mask, 0)
        x.masked_fill_(mask, 0)
        data[data > 2] = 0
        self.assertEqual(x, data)
        self.assertEqual(y, data)

    def test_matmul(self):
        test_shapes = [((2,), (2,)),
                       ((2,), (2, 3)),
                       ((2, 3), (3,)),
                       ((2, 3), (3, 4)),
                       ((2,), (4, 2, 3)),
                       ((4, 2, 3), (3,)),
                       ((1, 2, 3), (2, 3, 4)),
                       ((2, 2, 3), (1, 3, 4)),
                       ((2, 2, 3), (2, 3, 4)),
                       ((2, 1, 2, 3), (2, 3, 4)),
                       ((1, 2, 3), (2, 2, 3, 4)),
                       ((2, 1, 2, 3), (1, 2, 3, 4))]
        for a_shape, b_shape in test_shapes:
            data1, data2 = arange(a_shape), arange(b_shape, 1)
            a, b = new_tensor(data1, False), new_tensor(data2, False)
            self.assertEqual(a.__matmul__(b), np.matmul(data1, data2))

    def test_max(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for axis, keepdims in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = x.max(axis, keepdim=keepdims)
            result = np.max(data, axis, keepdims=keepdims)
            self.assertEqual(y, result)

    def test_maximum(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = uniform(a_shape), uniform(b_shape)
            a, b = new_tensor(data1), new_tensor(data2)
            y = a.maximum(b)
            self.assertEqual(y, np.maximum(data1, data2))

    def test_mean(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((1, 2), True), ((1, 2), False)]
        for axis, keepdims in entries:
            data = arange((2, 3, 3))
            x = new_tensor(data)
            y = x.mean(axis, keepdim=keepdims)
            result = np.mean(data, axis, keepdims=keepdims)
            self.assertEqual(y, result)

    def test_min(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for axis, keepdims in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = x.min(axis, keepdim=keepdims)
            result = np.min(data, axis, keepdims=keepdims)
            self.assertEqual(y, result)

    def test_minimum(self):
        for a_shape, b_shape in self.binary_test_shapes:
            data1, data2 = uniform(a_shape), uniform(b_shape)
            a, b = new_tensor(data1), new_tensor(data2)
            y = a.minimum(b)
            self.assertEqual(y, np.minimum(data1, data2))

    def test_mm(self):
        entries = [((2, 3), (3, 4))]
        for a_shape, b_shape in entries:
            data1, data2 = arange(a_shape), arange(b_shape)
            a, b = new_tensor(data1), new_tensor(data2)
            y = a.mm(b)
            self.assertEqual(y, np.matmul(data1, data2))

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
            self.assertEqual(a != b, np.not_equal(data1, data2))

    def test_neg(self):
        data = np.array([-1., 0., 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(-x, -data)
        x.neg_()
        self.assertEqual(x, -data)

    def test_non_zero(self):
        data = arange((2, 3))
        x = new_tensor(data)
        self.assertEqual((x > 2).nonzero(), np.stack(np.nonzero(data > 2), axis=1))

    def test_norm(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for axis, keepdims in entries:
            for p in (1, 2, 'fro', None):
                data = arange((2, 3))
                x = new_tensor(data)
                y = x.norm(p, axis, keepdim=keepdims)
                if p == 1:
                    result = np.sum(np.abs(data), axis=axis, keepdims=keepdims)
                elif p == 2 or p == 'fro':
                    result = np.sum(np.square(data), axis=axis, keepdims=keepdims)
                    result = np.sqrt(result)
                else:
                    result = np.linalg.norm(data, p, axis, keepdims=keepdims)
                self.assertEqual(y, result)

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
                self.assertEqual(x.T, data.T)
                x.permute_()
                self.assertEqual(x, np.transpose(data))
            else:
                self.assertEqual(x.permute(*perm), np.transpose(data, perm))
                x.permute_(*perm)
                self.assertEqual(x, np.transpose(data, perm))
        entries = [(0, 1), (0, 2), (1, 2)]
        for dim0, dim1 in entries:
            data = arange((2, 3, 4))
            x = new_tensor(data)
            perm = list(range(len(data.shape)))
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
            self.assertEqual(x.transpose(dim0, dim1), np.transpose(data, perm))
            x.transpose_(dim0, dim1)
            self.assertEqual(x, np.transpose(data, perm))

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
            repeats = (1,) * (len(data.shape) - len(repeats)) + repeats
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

    def test_roll(self):
        entries = [(0, 0), ((0, 0), (0, 1)), ((-1, 1), (0, 1)), (1, None)]
        for shift, axis in entries:
            data = arange((2, 3))
            x = new_tensor(data)
            y = x.roll(shift, axis)
            self.assertEqual(y, np.roll(data, shift, axis))

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

    def test_scatter(self):
        for axis in range(0, 1):
            data1 = arange((4, 4))
            data2 = np.array([[0, 1, 2, 3], [1, 2, 3, 0],
                              [2, 3, 0, 1], [3, 0, 1, 2]], 'int64')
            data3 = arange((4, 4), 100)
            x, index = new_tensor(data1), new_tensor(data2)
            v = new_tensor(data3)
            y = x.scatter(axis, index, v)
            result = data1.copy()
            for i, j in itertools.product(*[range(d) for d in data2.shape]):
                if axis == 0:
                    result[data2[i, j], j] = data3[i, j]
                else:
                    result[i, data2[i, j]] = data3[i, j]
            self.assertEqual(y, result)
            x.scatter_(axis, index, v)
            self.assertEqual(x, result)

    def test_scatter_add(self):
        for axis in range(0, 1):
            data1 = arange((4, 4))
            data2 = np.array([[0, 0], [0, 0]], 'int64')
            data3 = arange((4, 4), 100)
            x, index = new_tensor(data1), new_tensor(data2)
            v = new_tensor(data3)
            y = x.scatter_add(axis, index, v)
            result = data1.copy()
            for i, j in itertools.product(*[range(d) for d in data2.shape]):
                if axis == 0:
                    result[data2[i, j], j] += data3[i, j]
                else:
                    result[i, data2[i, j]] += data3[i, j]
            self.assertEqual(y, result)
            x.scatter_(axis, index, v, reduce='add')
            self.assertEqual(x, result)

    def test_scatter_mul(self):
        for axis in range(0, 1):
            data1 = arange((4, 4))
            data2 = np.array([[0, 1, 2, 3], [1, 2, 3, 0],
                              [2, 3, 0, 1], [3, 0, 1, 2]], 'int64')
            x, index = new_tensor(data1), new_tensor(data2)
            result = data1.copy()
            for i, j in itertools.product(*[range(d) for d in data2.shape]):
                if axis == 0:
                    result[data2[i, j], j] *= 2.33
                else:
                    result[i, data2[i, j]] *= 2.33
            x.scatter_(axis, index, 2.33, reduce='multiply')
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

    def test_sigmoid(self):
        data = np.array([0.2, 0.4, 0.6, 0.8, 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.sigmoid(), 1. / (1. + np.exp(-data)))
        x.sigmoid_()
        self.assertEqual(x, 1. / (1. + np.exp(-data)))

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

    def test_sort(self):
        entries = [(None, True),
                   (0, True),
                   (-1, True),
                   (0, False),
                   (-1, False)]
        for axis, descending in entries:
            data = uniform((5, 10))
            x = new_tensor(data)
            val, idx1 = x.sort(axis, descending)
            idx2 = x.argsort(axis, descending)
            axis = axis if axis is not None else -1
            result_val = np.sort(-data if descending else data, axis=axis)
            result_val = -result_val if descending else result_val
            result_idx = np.argsort(-data if descending else data, axis=axis)
            result_idx = np.take(result_idx, np.arange(data.shape[axis]), axis=axis)
            self.assertEqual(val, result_val)
            self.assertEqual(idx1, result_idx)
            self.assertEqual(idx2, result_idx)

    def test_split(self):
        entries = [((2, 4), 2, 1),
                   ((2, 3), 2, 1),
                   ((2, 3), (2, 1), 1)]
        for shape, size_or_sections, dim in entries:
            data = arange(shape)
            x = new_tensor(data)
            y = x.split(size_or_sections, dim)
            self.assertEqual(y, np.split(data, (2,), axis=1))

    def test_sqrt(self):
        data = np.array([4., 9., 16], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.sqrt(), np.sqrt(data))
        x.sqrt_()
        self.assertEqual(x, np.sqrt(data))

    def test_square(self):
        data = np.array([2., 3., 4], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.square(), np.square(data))

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

    def test_sum(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((1, 2), True), ((1, 2), False)]
        for axis, keepdims in entries:
            data = arange((2, 3, 3))
            x = new_tensor(data)
            y = x.sum(axis, keepdim=keepdims)
            result = np.sum(data, axis, keepdims=keepdims)
            self.assertEqual(y, result)

    def test_tanh(self):
        data = np.array([0.2, 0.4, 0.6, 0.8, 1.], 'float32')
        x = new_tensor(data)
        self.assertEqual(x.tanh(), np.tanh(data))
        x.tanh_()
        self.assertEqual(x, np.tanh(data))

    def test_tril(self):
        entries = [(3, 3), (3, 4,), (4, 3), (2, 3, 3)]
        for shape in entries:
            data = arange(shape, 1)
            for k in range(-max(shape), max(shape) + 1):
                x = new_tensor(data)
                y = x.tril(k)
                self.assertEqual(y, np.tril(data, k))
                x.tril_(k)
                self.assertEqual(x, np.tril(data, k))

    def test_triu(self):
        entries = [(3, 3), (3, 4,), (4, 3), (2, 3, 3)]
        for shape in entries:
            data = arange(shape, 1)
            for k in range(-max(shape), max(shape) + 1):
                x = new_tensor(data)
                y = x.triu(k)
                self.assertEqual(y, np.triu(data, k))
                x.triu_(k)
                self.assertEqual(x, np.triu(data, k))

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
            y = x.type(dtype)
            self.assertEqual(y.type(), dtype)

    def test_unbind(self):
        entries = [0, 1]
        for axis in entries:
            data = arange((2, 3))
            num = data.shape[axis]
            grad = np.ones(data.shape, 'float32')
            grad[tuple(slice(0, 1) if i == axis else
                       slice(None) for i in range(data.ndim))] = 0
            x = new_tensor(data)
            y = x.unbind(axis)
            result = [x.squeeze(axis) for x in np.split(data, num, axis)]
            self.assertEqual(y, result)

    def test_uniform(self):
        data = arange((2, 3))
        x = new_tensor(data)
        x.uniform_()

    def test_unique(self):
        data = np.array([1, 1, 3, 5, 5, 7, 9])
        entries = [(False, False),
                   (True, False),
                   (False, True),
                   (True, True)]
        for return_inverse, return_counts in entries:
            x = new_tensor(data)
            y = x.unique(return_inverse=return_inverse,
                         return_counts=return_counts,
                         sorted=True)
            result = np.unique(
                data,
                return_inverse=return_inverse,
                return_counts=return_counts)
            self.assertEqual(y, result)

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

    def test_var(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((1, 2), True), ((1, 2), False)]
        for axis, keepdims in entries:
            data = arange((2, 3, 3))
            x = new_tensor(data)
            y = x.var(axis, keepdim=keepdims)
            result = np.var(data, axis, keepdims=keepdims)
            self.assertEqual(y, result)


class TestTorchOps(OpTestCase):
    """Test builtin torch ops."""

    def test_arange(self):
        entries = [([5], {'dtype': 'int64'}),
                   ([0, 5], {'dtype': 'int64'}),
                   ([0, 5, 2], {'dtype': 'int64'}),
                   ([0., 1., 0.2], {'dtype': 'float32'})]
        for (args, kwargs) in entries:
            data = np.arange(*args, **kwargs)
            x = torch.arange(*args, **kwargs)
            self.assertEqual(x, data)

    def test_cat(self):
        entries = [0, 1]
        for axis in entries:
            data = arange((2, 2))
            x = new_tensor(data)
            y = torch.cat([x, x], dim=axis)
            self.assertEqual(y, np.concatenate([data, data], axis=axis))

    def test_linspace(self):
        entries = [([[0., 5.], [10., 40.], 5], {'dim': 0, 'dtype': 'float32'}),
                   ([[0., 5.], [10., 40.], 5], {'dim': 1, 'dtype': 'float32'}),
                   ([[0., 5.], [10., 40.], 5], {'dim': -1, 'dtype': 'float32'}),
                   ([[0.], [10.], 5], {'dim': 0, 'dtype': 'float32'}),
                   ([[0.], [10.], 5], {'dim': -1, 'dtype': 'float32'}),
                   ([0., 10., 5], {'dim': 0, 'dtype': 'float32'}),
                   ([0., 10., 5], {'dim': 0, 'dtype': 'int64'})]
        for (args, kwargs) in entries:
            x = torch.linspace(*args, **kwargs)
            kwargs['axis'] = kwargs.pop('dim')
            data = np.linspace(*args, **kwargs)
            self.assertEqual(x, data)

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

    def test_stack(self):
        entries = [0, 1]
        for axis in entries:
            data = arange((2, 2))
            x = new_tensor(data)
            y = torch.stack([x, x], dim=axis)
            self.assertEqual(y, np.stack([data, data], axis=axis))

    def test_var_mean(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((1, 2), True), ((1, 2), False)]
        for axis, keepdims in entries:
            data = arange((2, 3, 3))
            x = new_tensor(data)
            y = torch.var_mean(x, axis, keepdim=keepdims)
            result1 = np.var(data, axis, keepdims=keepdims)
            result2 = np.mean(data, axis, keepdims=keepdims)
            self.assertEqual(y, [result1, result2])

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
