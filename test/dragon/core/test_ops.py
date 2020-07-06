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
import os
import unittest

import dragon
import numpy as np

from dragon.core.eager.context import context as execution_context
from dragon.core.util import nest
from dragon.core.testing.unittest.common_utils import run_tests
from dragon.core.testing.unittest.common_utils import TEST_CUDA

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
        test_symbols=True,
    ):
        if prec is None:
            prec = self.precision
        inputs = nest.flatten(first)
        num_first = len(inputs)
        inputs += nest.flatten(second)
        num_second = len(inputs) - num_first
        symbols = []
        for i, input in enumerate(inputs):
            if isinstance(input, dragon.EagerTensor):
                inputs[i] = input.numpy()
            elif isinstance(input, dragon.Tensor):
                symbols.append((i, input))
        if len(symbols) > 0:
            values = nest.flatten(dragon.create_function(
                outputs=[symbol[1] for symbol in symbols])())
            for i, value in enumerate(values):
                if test_symbols:
                    dtype = symbols[i][1].dtype
                    shape = symbols[i][1].shape
                    super(OpTestCase, self).assertEqual(dtype, str(values[i].dtype))
                    super(OpTestCase, self).assertEqual(shape, list(shape))
                inputs[symbols[i][0]] = values[i]
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
                self.assertEqual(a, b, msg, prec, test_symbols)
        else:
            super(OpTestCase, self).assertEqual(first, second, msg)


class TestActivationOps(OpTestCase):
    """Test the activation ops."""

    def __init__(self, method_name='runTest'):
        super(TestActivationOps, self).__init__(method_name)
        self.cudnn_ws = dragon.Workspace()

    def test_dropout(self):
        prob = 0.
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = uniform((2, 3))
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.nn.dropout(x, prob=prob)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [data, data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_dropout_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_dropout()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_dropout_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_dropout()

    def test_drop_block2d(self):
        keep_prob, block_size = 1., 2
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for data_format in ('NCHW', 'NHWC'):
                    data = uniform((2, 3, 4, 4) if data_format == 'NCHW'
                                   else (2, 4, 4, 3))
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.nn.drop_block2d(
                            x,
                            block_size=block_size,
                            keep_prob=keep_prob,
                            data_format=data_format)
                    dx = tape.gradient(y, [x], output_gradients=[x])[0]
                    self.assertEqual([y, dx], [data, data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_drop_block2d_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_drop_block2d()

    def test_drop_path(self):
        prob = 0.
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = uniform((2, 3))
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.nn.drop_path(x, prob=prob)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [data, data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_drop_path_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_drop_path()

    def test_elu(self):
        alpha = 1.
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.nn.elu(x, alpha)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = np.maximum(data, 0.) + alpha * (np.exp(np.minimum(data, 0.)) - 1.)
                self.assertEqual(
                    [y, dx], [result, data * ((data > 0.) + (data < 0.) * (alpha + result))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_elu_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_elu()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_elu_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_elu()

    def test_leaky_relu(self):
        alpha = 0.2
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.nn.leaky_relu(x, alpha)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = np.maximum(data, 0.) + np.minimum(data, 0.) * alpha
                self.assertEqual([y, dx], [result, data * ((data > 0.) + (data < 0.) * alpha)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_leaky_relu_cuda(self):
        with dragon.device('cuda'):
            self.test_leaky_relu()

    def test_log_softmax(self):
        grad = np.array([[-0.90813, -0.15201, 1.06013],
                         [-1.87572, 2.63141, -0.7557]], dtype='float32')
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data1 = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32')
                data2 = np.log(data1)
                x = new_tensor(data2)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.nn.log_softmax(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [data2, grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_log_softmax_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_log_softmax()

    def test_prelu(self):
        entries = [((3,), (1,), 'NCHW'),
                   ((3,), (3,), 'NCHW'),
                   ((2, 3), (1,), 'NCHW'),
                   ((2, 3), (3,), 'NCHW'),
                   ((2, 3, 4, 4), (1,), 'NCHW'),
                   ((2, 3, 4, 4), (1, 3, 1, 1), 'NCHW'),
                   ((2, 4, 4, 3), (1,), 'NHWC'),
                   ((2, 4, 4, 3), (1, 1, 1, 3), 'NHWC')]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for x_shape, w_shape, data_format in entries:
                    data1 = uniform(x_shape)
                    data2 = np.ones(w_shape, 'float32') * 0.25
                    x, w = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w])
                        y = dragon.nn.prelu([x, w], data_format=data_format)
                    dx, dw = tape.gradient(y, [x, w], output_gradients=[x])
                    result = np.maximum(data1, 0.) + np.minimum(data1, 0.) * data2
                    grad1 = data1 * ((data1 > 0.) + (data1 < 0.) * data2)
                    grad2 = reduce_like(data1 * ((data1 < 0.) * data1), data2)
                    self.assertEqual([y, dx, dw], [result, grad1, grad2.reshape((-1,))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_prelu_cuda(self):
        with dragon.device('cuda'):
            self.test_prelu()

    def test_relu(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.nn.relu(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = np.maximum(data, 0.)
                self.assertEqual([y, dx], [result, data * (data > 0.)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_relu_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_relu()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_relu_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_relu()

    def test_relu6(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1., 6., 7.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.nn.relu6(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = np.minimum(np.maximum(data, 0.), 6.)
                self.assertEqual(
                    [y, dx], [result, data * (np.bitwise_and(data > 0, data < 6))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_relu6_cuda(self):
        with dragon.device('cuda'):
            self.test_relu()

    def test_selu(self):
        alpha, gamma = 1.67326, 1.0507
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.nn.selu(x, alpha, gamma)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = gamma * (
                    np.maximum(data, 0.) +
                    alpha * (np.exp(np.minimum(data, 0.)) - 1.))
                self.assertEqual(
                    [y, dx],
                    [result, data * ((data > 0.) * gamma +
                                     (data < 0.) * (alpha * gamma + result))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_selu_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_selu()

    def test_sigmoid(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([0.2, 0.4, 0.6, 0.8, 1.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.sigmoid(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = 1. / (1. + np.exp(-data))
                self.assertEqual([y, dx], [result, data * result * (1. - result)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sigmoid_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_sigmoid()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sigmoid_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_sigmoid()

    def test_softmax(self):
        grad = np.array([[-0.11596, -0.0523, 0.16825],
                         [-0.15008, 0.3116, -0.16152]], dtype='float32')
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32')
                x = new_tensor(np.log(data))
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.nn.softmax(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [data, grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_softmax_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_softmax()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_softmax_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_softmax()

    def test_tanh(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([0.2, 0.4, 0.6, 0.8, 1.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.tanh(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = np.tanh(data)
                self.assertEqual([y, dx], [result, data * (1. - np.square(result))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_tanh_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_tanh()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_tanh_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_tanh()


class TestArrayOps(OpTestCase):
    """Test the array ops."""

    def test_arange(self):
        entries = [([5], {'dtype': 'int64'}),
                   ([0, 5], {'dtype': 'int64'}),
                   ([0, 5, 2], {'dtype': 'int64'}),
                   ([0., 1., 0.2], {'dtype': 'float32'})]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (args, kwargs) in entries:
                    data = np.arange(*args, **kwargs)
                    x = dragon.arange(*args, **kwargs)
                    self.assertEqual(x, data)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_arange_cuda(self):
        with dragon.device('cuda'):
            self.test_arange()

    def test_broadcast_to(self):
        entries = [((2, 2, 3, 1), (0, True)),
                   ((1, 2, 3, 2), (3, True)),
                   ((2, 2, 3, 2), ((0, 3), True)),
                   ((2, 1, 2, 3, 1), (0, False))]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for shape, (axis, keepdims) in entries:
                    expand_size = int(np.prod(shape))
                    data = np.arange(6).astype('float32').reshape((1, 2, 3, 1))
                    grad = np.arange(expand_size).astype('float32').reshape(shape)
                    x = new_tensor(data)
                    dy = new_tensor(grad)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.broadcast_to(x, shape)
                    dx = tape.gradient(y, x, output_gradients=[dy])[0]
                    self.assertEqual(
                        [y, dx], [np.broadcast_to(data, shape),
                                  grad.sum(axis=axis, keepdims=keepdims)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_broadcast_to_cuda(self):
        with dragon.device('cuda'):
            self.test_broadcast_to()

    def test_cast(self):
        entries = [('int8', 'uint8'),
                   ('int32', 'float32'),
                   ('float32', 'int32'),
                   ('float32', 'float16'),
                   ('float32', 'float32'),
                   ('float32', 'float64')]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for in_type, out_type in entries:
                    data1 = np.array([-2., -1., 0., 1., 2.], dtype=in_type)
                    data2 = data1.astype(out_type)
                    x, dy = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.cast(x, dtype=out_type)
                    dx = y if in_type == out_type \
                        else tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [data2, data1])
                    dragon.cast(x, dtype=out_type, inplace=True)
                    self.assertEqual(x, data2)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_cast_cuda(self):
        with dragon.device('cuda'):
            self.test_cast()

    def test_channel_normalize(self):
        entries = [((2, 3, 4), [(1., 2., 3.), (3., 2., 1.), 1], {'perm': (0, 1, 2)}),
                   ((2, 3, 4), [(1., 2., 3.), (3., 2., 1.), 2], {'perm': (0, 2, 1)})]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for shape, args, kwargs in entries:
                    perm = kwargs['perm']
                    data = np.ones(shape, dtype='uint8').transpose(perm)
                    mean = np.array(args[0]).reshape((1, 3, 1)).transpose(perm)
                    std = np.array(args[1]).reshape((1, 3, 1)).transpose(perm)
                    x = dragon.ones(shape, dtype='uint8')
                    y = dragon.channel_normalize(x, *args, **kwargs)
                    self.assertEqual(y, (data - mean) / std)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_channel_normalize_cuda(self):
        with dragon.device('cuda'):
            self.test_channel_normalize()

    def test_channel_shuffle(self):
        entries = [(0, 2), (1, 4)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, group in entries:
                    data = arange((2, 8))
                    g, k = group, data.shape[axis] // group
                    shape1 = data.shape[:axis] + (g, k) + data.shape[axis + 1:]
                    shape2 = data.shape[:axis] + (k, g) + data.shape[axis + 1:]
                    perm = list(range(0, axis)) + [axis + 1, axis] + list(range(axis + 2, len(shape1)))
                    x, dy = new_tensor(data), new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.channel_shuffle(x, axis, group)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual(
                        [y, dx], [data.reshape(shape1).transpose(perm).reshape(data.shape),
                                  data.reshape(shape2).transpose(perm).reshape(data.shape)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_channel_shuffle_cuda(self):
        with dragon.device('cuda'):
            self.test_channel_shuffle()

    def test_concat(self):
        entries = [0, 1]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis in entries:
                    data = arange((2, 2))
                    grad = np.concatenate([data, data], axis=axis)
                    x = new_tensor(data)
                    dy = new_tensor(grad)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.concat([x, x], axis=axis)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual(
                        [y, dx], [np.concatenate([data, data], axis=axis),
                                  sum(np.split(grad, 2, axis=axis))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_concat_cuda(self):
        with dragon.device('cuda'):
            self.test_concat()

    def test_expand_dims(self):
        entries = [1, -1]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis in entries:
                    data = arange((2, 3, 4))
                    grad = np.expand_dims(data, axis=axis)
                    x = new_tensor(data)
                    dy = new_tensor(grad)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.expand_dims(x, axis=axis)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [grad, data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_expand_dims_cuda(self):
        with dragon.device('cuda'):
            self.test_expand_dims()

    def test_flatten(self):
        entries = [(-2, -1, None), (1, -1, None), (1, 2, None), (0, -1, 1), (0, -1, 2)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, num_axes, keep_axes in entries:
                    data = arange((1, 2, 3, 4))
                    if keep_axes is not None:
                        new_shape = data.shape[:keep_axes - 1] + \
                            (int(np.prod(data.shape[keep_axes - 1:])),)
                    else:
                        if axis < 0:
                            axis += len(data.shape)
                        if num_axes < 0:
                            num_axes = len(data.shape) - axis
                        new_shape = \
                            data.shape[:axis] + \
                            (int(np.prod(data.shape[axis:axis + num_axes])),) + \
                            data.shape[axis + num_axes:]
                    grad = data.reshape(new_shape)
                    x = new_tensor(data)
                    dy = new_tensor(grad)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.flatten(x, axis, num_axes, keep_axes)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [grad, data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_flatten_cuda(self):
        with dragon.device('cuda'):
            self.test_flatten()

    def test_index_select(self):
        entries = [1, (1, 2)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
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
                    grad = np.zeros((1, 2, 3, 4), 'float32').reshape(flatten_shape)
                    for i in index:
                        slices = [slice(None, None, None)] * (len(flatten_shape) - 1)
                        slices.insert(axes[0], i)
                        grad[tuple(slices)] += 1
                    x = new_tensor(data)
                    x_index = new_tensor(index)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.index_select(x, x_index, axis=axes)
                    dx = tape.gradient(y, [x])[0]
                    self.assertEqual(
                        [y, dx],
                        [np.take(data.reshape(flatten_shape), index, axis=axes[0]),
                         grad.reshape(data.shape)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_index_select_cuda(self):
        with dragon.device('cuda'):
            self.test_index_select()

    def test_masked_select(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = arange((2, 3))
                grad = np.zeros((2, 3), dtype='float32')
                grad[data > 2] = 1
                grad *= data
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.masked_select([x, x > 2])
                dx = tape.gradient(y, [x], output_gradients=[y])[0]
                self.assertEqual([y, dx], [data[data > 2], grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_masked_select_cuda(self):
        with dragon.device('cuda'):
            self.test_masked_select()

    def test_non_zero(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = arange((2, 3))
                x = new_tensor(data)
                y = dragon.nonzero(x > 2)
                self.assertEqual(y, np.stack(np.nonzero(data > 2), axis=1))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_non_zero_cuda(self):
        with dragon.device('cuda'):
            self.test_non_zero()

    def test_one_hot(self):
        entries = [(2, 3, 3), (1, 2, 3), (2, 2, 2)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for index in entries:
                    index = np.array(index, 'int64')
                    x = new_tensor(index)
                    y = dragon.one_hot(x, depth=10)
                    self.assertEqual(y, np.eye(10, dtype='int64')[index])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_one_hot_cuda(self):
        with dragon.device('cuda'):
            self.test_one_hot()

    def test_pad(self):
        entries = [([(0, 1)], 'constant'),
                   ([(1, 1)], 'reflect'),
                   ([(2, 1)], 'edge')]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for pads, mode in entries:
                    data = arange((6,))
                    grad = np.ones((6,), 'float32') if mode == 'constant' else None
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.pad(x, pads, mode)
                    dx = tape.gradient(y, [x])[0] if mode == 'constant' else None
                    self.assertEqual([y, dx], [np.pad(data, pads, mode), grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_pad_cuda(self):
        with dragon.device('cuda'):
            self.test_pad()

    def test_repeat(self):
        entries = [(None, 2), (1, 2)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, repeats in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.repeat(x, axis, repeats)
                    grad = arange(y.shape)
                    grad_shape = y.shape[:-1] + [y.shape[-1] // 2, 2]
                    dy = new_tensor(grad)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual(
                        [y, dx], [np.repeat(data, repeats, axis),
                                  grad.reshape(grad_shape).sum(-1).reshape(data.shape)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_repeat_cuda(self):
        with dragon.device('cuda'):
            self.test_repeat()

    def test_reshape(self):
        entries = [(0, 0), (0, -1)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for shape in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.reshape(x, shape)
                    grad = data.reshape(y.shape)
                    dy = new_tensor(grad)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [grad, data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_reshape_cuda(self):
        with dragon.device('cuda'):
            self.test_reshape()

    def test_shape(self):
        entries = [(2, 3), (2, 3, 3)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for shape in entries:
                    self.assertEqual(dragon.shape(dragon.ones(shape)),
                                     np.array(shape, 'int64'))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_shape_cuda(self):
        with dragon.device('cuda'):
            self.test_shape()

    def test_slice(self):
        entries = [0,
                   slice(0, None, None),
                   (slice(None, None, None), slice(1, None, None)),
                   (slice(None, None, None),) * 3]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for item in entries:
                    data = arange((1, 2, 3))
                    grad = np.zeros_like(data, 'float32')
                    grad.__setitem__(item, 1.)
                    grad *= data
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.slice(x, *process_indices(item))
                    dx = tape.gradient(y, [x], output_gradients=[y])[0]
                    self.assertEqual([y, dx], [data.__getitem__(item), grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_slice_cuda(self):
        with dragon.device('cuda'):
            self.test_slice()

    def test_split(self):
        entries = [(2, 1, None), ((2, 1), 1, None), (2, 1, (2,))]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for args in entries:
                    data = arange((2, 3))
                    grad = np.ones((2, 3), 'float32')
                    grad[:, -1] = 0
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.split(x, *args)
                    dx = tape.gradient(y[0], [x])[0]
                    self.assertEqual([y, dx], [np.split(data, (2,), axis=1), grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_split_cuda(self):
        with dragon.device('cuda'):
            self.test_split()

    def test_squeeze(self):
        entries = [((2, 1, 3), 1), ((1, 2, 1, 3), (0, 2)), ((3, 1, 2, 1), (1,))]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for shape, axis in entries:
                    data = arange(shape)
                    grad = np.squeeze(data, axis)
                    x = new_tensor(data)
                    dy = new_tensor(grad)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.squeeze(x, axis)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [grad, data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_squeeze_cuda(self):
        with dragon.device('cuda'):
            self.test_squeeze()

    def test_stack(self):
        entries = [0, 1]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis in entries:
                    data = arange((2, 2))
                    grad = np.stack([data, data], axis=axis)
                    x = new_tensor(data)
                    dy = new_tensor(grad)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.stack([x, x], axis=axis)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual(
                        [y, dx], [np.stack([data, data], axis=axis),
                                  sum(np.split(grad, 2, axis=axis)).squeeze()])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_stack_cuda(self):
        with dragon.device('cuda'):
            self.test_stack()

    def test_tile(self):
        entries = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for multiples in entries:
                    data = arange((2, 2))
                    grad = np.tile(data, multiples)
                    x = new_tensor(data)
                    dy = new_tensor(grad)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.tile(x, multiples)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [grad, data * np.prod(multiples)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_tile_cuda(self):
        with dragon.device('cuda'):
            self.test_tile()

    def test_transpose(self):
        entries = [(0, 2, 1), None]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for perm in entries:
                    data1 = arange((2, 3, 4))
                    data2 = np.transpose(data1, perm)
                    x, dy = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.transpose(x, perm)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [data2, data1])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_transpose_cuda(self):
        with dragon.device('cuda'):
            self.test_transpose()

    def test_where(self):
        entries = [((6,), (6,))]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in entries:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    data3 = data1 > 1
                    grad1 = np.zeros(a_shape, 'float32')
                    grad2 = np.zeros(b_shape, 'float32')
                    grad1[data1 > 1] = 1
                    grad2[data1 <= 1] = 1
                    a, b, c = new_tensor(data1), new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.where([a, b, c])
                    da, db = tape.gradient(y, [a, b], output_gradients=[y])
                    result = np.where(data3, data1, data2)
                    self.assertEqual(
                        [y, da, db],
                        [result, result * grad1, result * grad2])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_where_cuda(self):
        with dragon.device('cuda'):
            self.test_where()


class TestControlFlowOps(OpTestCase):
    """Test the control flow ops."""

    def test_assign(self):
        entries = [0,
                   slice(0, None, None),
                   (slice(None, None, None), slice(1, None, None)),
                   (slice(None, None, None),) * 3]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for item in entries:
                    data = np.zeros((1, 2, 3), dtype='float32')
                    data.__setitem__(item, 1.)
                    x = new_tensor(data)
                    dragon.assign([x, 1.], *process_indices(item))
                    self.assertEqual(x, data)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_assign_cuda(self):
        with dragon.device('cuda'):
            self.test_assign()

    def test_copy(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = arange((4,))
                x = new_tensor(data)
                y = dragon.zeros((4,), dtype='float32')
                y = dragon.copy([x, y])
                z = dragon.copy(x)
                self.assertEqual(y, data)
                self.assertEqual(z, data)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_copy_cuda(self):
        with dragon.device('cuda'):
            self.test_copy()

    def test_masked_assign(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = arange((2, 3))
                x = new_tensor(data)
                dragon.masked_assign([x, 0, x > 2])
                data[data > 2] = 0
                self.assertEqual(x, data)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_masked_assign_cuda(self):
        with dragon.device('cuda'):
            self.test_masked_assign()


class TestInitOps(OpTestCase):
    """Test the init ops."""

    def test_eye(self):
        entries = [(2,), (2, 3), (2, 3, 2), (2, 3, 3), (2, 3, -2)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for args in entries:
                    x = dragon.eye(*args, dtype='float32')
                    self.assertEqual(x, np.eye(*args, dtype='float32'))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_eye_cuda(self):
        with dragon.device('cuda'):
            self.test_eye()

    def test_eye_like(self):
        entries = [((2, 3), 0)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for shape, k in entries:
                    data = np.zeros(shape)
                    x_ref = new_tensor(data)
                    x = dragon.eye_like(x_ref, k=k, dtype='float32')
                    self.assertEqual(x, np.eye(*shape, k=k, dtype='float32'))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_eye_like_cuda(self):
        with dragon.device('cuda'):
            self.test_eye_like()

    def test_fill(self):
        entries = [((2, 3), 0)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for shape, value in entries:
                    data = np.zeros(shape, dtype='float32')
                    data.fill(value)
                    x = dragon.fill(shape, value=value, dtype='float32')
                    self.assertEqual(x, data)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_fill_cuda(self):
        with dragon.device('cuda'):
            self.test_fill()

    def test_glorot_normal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                dragon.random.glorot_normal((2, 3))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_glorot_normal_cuda(self):
        with dragon.device('cuda'):
            self.test_glorot_normal()

    def test_glorot_uniform(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                dragon.random.glorot_uniform((2, 3))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_glorot_uniform_cuda(self):
        with dragon.device('cuda'):
            self.test_glorot_uniform()

    def test_ones(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                self.assertEqual(dragon.ones((2, 3), 'float32'),
                                 np.ones((2, 3), 'float32'))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_ones_cuda(self):
        with dragon.device('cuda'):
            self.test_ones()

    def test_ones_like(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.ones((2, 3), 'float32')
                x_ref = new_tensor(data)
                x = dragon.ones_like(x_ref, 'float32')
                self.assertEqual(x, data)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_ones_like_cuda(self):
        with dragon.device('cuda'):
            self.test_ones_like()

    def test_random_normal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                dragon.random.normal((2, 3))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_random_normal_cuda(self):
        with dragon.device('cuda'):
            self.test_random_normal()

    def test_random_normal_like(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.zeros((2, 3), 'float32')
                x_ref = new_tensor(data)
                dragon.random.normal_like(x_ref)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_random_normal_like_cuda(self):
        with dragon.device('cuda'):
            self.test_random_normal_like()

    def test_random_uniform(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                dragon.random.uniform((2, 3))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_random_uniform_cuda(self):
        with dragon.device('cuda'):
            self.test_random_uniform()

    def test_random_uniform_like(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.zeros((2, 3), 'float32')
                x_ref = new_tensor(data)
                dragon.random.uniform_like(x_ref)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_random_uniform_like_cuda(self):
        with dragon.device('cuda'):
            self.test_random_uniform_like()

    def test_truncated_normal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                dragon.random.truncated_normal((2, 3))

    def test_zeros(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                self.assertEqual(dragon.zeros((2, 3), dtype='float32'),
                                 np.zeros((2, 3), dtype='float32'))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_zeros_cuda(self):
        with dragon.device('cuda'):
            self.test_zeros()

    def test_zeros_like(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.zeros((2, 3), dtype='float32')
                x_ref = new_tensor(data)
                x = dragon.zeros_like(x_ref, dtype='float32')
                self.assertEqual(x, data)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_zeros_like_cuda(self):
        with dragon.device('cuda'):
            self.test_zeros_like()


class TestLossOps(OpTestCase):
    """Test the loss ops."""

    def test_l1_loss(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for reduction in ('mean', 'sum', 'none'):
                    data1 = np.array([-1., 0., 1.], 'float32')
                    data2 = np.array([1., 0., -1.], 'float32')
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.losses.l1_loss([a, b], reduction)
                    data3 = arange(y.shape, 2)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    scale = np.sign(data1 - data2) / \
                        (data1.size if reduction == 'mean' else 1)
                    result = reduce(np.abs(data1 - data2), reduction=reduction)
                    self.assertEqual([y, da, db], [result, data3 * scale, data3 * (-scale)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_l1_loss_cuda(self):
        with dragon.device('cuda'):
            self.test_l1_loss()

    def test_l2_loss(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for reduction in ('mean', 'sum', 'none'):
                    data1 = np.array([-1., 0., 1.], 'float32')
                    data2 = np.array([1., 0., -1.], 'float32')
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.losses.l2_loss([a, b], reduction)
                    data3 = arange(y.shape, 2)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    scale = (data1 - data2) / \
                        (data1.size * 0.5 if reduction == 'mean' else 0.5)
                    result = reduce(np.square(data1 - data2), reduction=reduction)
                    self.assertEqual([y, da, db], [result, data3 * scale, data3 * (-scale)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_l2_loss_cuda(self):
        with dragon.device('cuda'):
            self.test_l2_loss()

    def test_nll_loss(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for reduction in ('mean', 'sum', 'none'):
                    data1 = np.log(np.array(
                        [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32'))
                    data2 = np.array([0, 1], 'int64')
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch(a)
                        y = dragon.losses.nll_loss([a, b], reduction=reduction)
                    data3 = arange(y.shape, 2)
                    dy = new_tensor(data3)
                    da = tape.gradient(y, [a], output_gradients=[dy])[0]
                    scale = (-np.ones_like(data1, 'float32') * np.eye(3)[data2]) / \
                        (data1.shape[0] if reduction == 'mean' else 1)
                    result = reduce(-data1[np.arange(2), data2], reduction=reduction)
                    self.assertEqual([y, da], [result, np.expand_dims(data3, -1) * scale])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_nll_loss_cuda(self):
        with dragon.device('cuda'):
            self.test_nll_loss()

    def test_sigmoid_cross_entropy(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for reduction in ('mean', 'sum', 'none'):
                    data1 = np.array([[0.2], [0.5], [0.7]], 'float32')
                    data2 = -np.log(1. / data1 - 1.)
                    data3 = np.array([[0], [1], [0]], 'float32')
                    a, b = new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch(a)
                        y = dragon.losses.sigmoid_cross_entropy(
                            [a, b], reduction=reduction)
                    data4 = arange(y.shape, 2)
                    dy = new_tensor(data4)
                    da = tape.gradient(y, [a], output_gradients=[dy])[0]
                    scale = (data1 - data3) / \
                        (data1.shape[0] if reduction == 'mean' else 1)
                    result = reduce(
                        -(data3 * np.log(data1) + (1 - data3) * np.log(1 - data1)),
                        reduction=reduction)
                    self.assertEqual([y, da], [result, data4 * scale])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sigmoid_cross_entropy_cuda(self):
        with dragon.device('cuda'):
            self.test_sigmoid_cross_entropy()

    def test_sigmoid_focal_loss(self):
        pos_alpha, neg_alpha, gamma = 0.25, 0.75, 2.0
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for reduction in ('mean', 'sum', 'none'):
                    data1 = np.array([[0.2, 0.3], [0.5, 0.1], [0.7, 0.2]], 'float32')
                    data2 = -np.log(1. / data1 - 1.)
                    data3 = np.array([0, 1, 0], 'int64')
                    a, b = new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch(a)
                        y = dragon.losses.sigmoid_focal_loss(
                            [a, b], reduction=reduction)
                    data4 = arange(y.shape, 2)
                    dy = new_tensor(data4)
                    da = tape.gradient(y, [a], output_gradients=[dy])[0]
                    pos_term = np.power((1. - data1), gamma) * np.log(data1)
                    pos_term *= (-pos_alpha * np.eye(2, dtype='float32')[data3])
                    neg_term = np.power(data1, gamma) * np.log(1. - data1)
                    neg_term *= (-neg_alpha * np.invert(np.eye(2, dtype='bool')[data3]))
                    result = reduce(pos_term + neg_term, reduction=reduction)
                    pos_term = np.power((1. - data1), gamma) * \
                        (1. - data1 - gamma * data1 * np.log(data1))
                    pos_term *= (-pos_alpha * np.eye(2, dtype='float32')[data3])
                    neg_term = np.power(data1, gamma) * \
                        (gamma * (1. - data1) * np.log(1. - data1) - data1)
                    neg_term *= (-neg_alpha * np.invert(np.eye(2, dtype='bool')[data3]))
                    grad = (data4 * (pos_term + neg_term)) / \
                        (data1.size if reduction == 'mean' else 1)
                    self.assertEqual([y, da], [result, grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sigmoid_focal_loss_cuda(self):
        with dragon.device('cuda'):
            self.test_sigmoid_focal_loss()

    def test_smooth_l1_loss(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for reduction in ('mean', 'sum', 'none'):
                    for beta in (1.,):
                        data1 = np.array([-1., 0., 1.], 'float32')
                        data2 = np.array([1., 0., 1.01], 'float32')
                        a, b = new_tensor(data1), new_tensor(data2)
                        with dragon.GradientTape() as tape:
                            tape.watch([a, b])
                            y = dragon.losses.smooth_l1_loss([a, b], beta, reduction)
                        data3 = arange(y.shape, 2)
                        dy = new_tensor(data3)
                        da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                        diff, abs_diff = data1 - data2, np.abs(data1 - data2)
                        scale = np.where(abs_diff < beta, diff / beta, np.sign(diff)) / \
                            (data1.size if reduction == 'mean' else 1)
                        result = reduce(np.where(
                            abs_diff < beta,
                            0.5 * np.square(diff) / beta,
                            abs_diff - 0.5 * beta),
                            reduction=reduction)
                        self.assertEqual([y, da], [result, data3 * scale])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_smooth_l1_loss_cuda(self):
        with dragon.device('cuda'):
            self.test_smooth_l1_loss()

    def test_softmax_cross_entropy(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for reduction in ('mean', 'sum', 'none'):
                    data1 = np.log(np.array(
                        [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32'))
                    data2 = np.array([0, 1], 'int64')
                    data3 = np.eye(3, dtype='float32')[data2]
                    a, b = new_tensor(data1), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch(a)
                        y = dragon.losses.softmax_cross_entropy(
                            [a, b], reduction=reduction)
                    data4 = arange(y.shape, 2)
                    dy = new_tensor(data4)
                    da = tape.gradient(y, [a], output_gradients=[dy])[0]
                    scale = (np.exp(data1) - data3) / \
                        (data1.shape[0] if reduction == 'mean' else 1)
                    result = reduce(-data1[np.arange(2), data2], reduction=reduction)
                    self.assertEqual([y, da], [result, np.expand_dims(data4, -1) * scale])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_softmax_cross_entropy_cuda(self):
        with dragon.device('cuda'):
            self.test_softmax_cross_entropy()

    def test_sparse_softmax_cross_entropy(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for reduction in ('mean', 'sum', 'none'):
                    data1 = np.log(np.array(
                        [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32'))
                    data2 = np.array([0, 1], 'int64')
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch(a)
                        y = dragon.losses.sparse_softmax_cross_entropy(
                            [a, b], reduction=reduction)
                    data3 = arange(y.shape, 2)
                    dy = new_tensor(data3)
                    da = tape.gradient(y, [a], output_gradients=[dy])[0]
                    scale = (np.exp(data1) - np.eye(3)[data2]) / \
                        (data1.shape[0] if reduction == 'mean' else 1)
                    result = reduce(-data1[np.arange(2), data2], reduction=reduction)
                    self.assertEqual([y, da], [result, np.expand_dims(data3, -1) * scale])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sparse_softmax_cross_entropy_cuda(self):
        with dragon.device('cuda'):
            self.test_sparse_softmax_cross_entropy()


class TestMathOps(OpTestCase):
    """Test the math ops."""

    # Testing shapes for binary ops
    unary_test_shapes = [(2,)]

    # Testing shapes for binary ops
    binary_test_shapes = [((2,), (2,)),
                          ((2, 3), (3,)),
                          ((2, 1), (2, 3)),
                          ((3,), (2, 3)),
                          ((2, 3), (2, 1)),
                          ((2, 1), (1, 3))]

    def test_abs(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.abs(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [np.abs(data), data * data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_abs_cuda(self):
        with dragon.device('cuda'):
            self.test_abs()

    def test_add(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.math.add([a, b])
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    self.assertEqual(
                        [y, da, db],
                        [data1 + data2,
                         reduce_like(data3, data1),
                         reduce_like(data3, data2)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_add_cuda(self):
        with dragon.device('cuda'):
            self.test_add()

    def test_affine(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data1 = arange((2, 3, 4, 5))
                data2, data3 = arange((3, 4)), arange((3, 4))
                data4 = arange(data1.shape)
                grad1 = data4 * np.expand_dims(data2, -1)
                grad2 = np.sum(data4 * data1, (0, 3))
                grad3 = np.sum(data4, (0, 3))
                x, w, b = new_tensor(data1), new_tensor(data2), new_tensor(data3)
                with dragon.GradientTape() as tape:
                    tape.watch([x, w, b])
                    y = dragon.math.affine([x, w, b], axis=1, num_axes=2)
                dy = new_tensor(data4)
                dx, dw, db = tape.gradient(y, [x, w, b], output_gradients=[dy])
                self.assertEqual(
                    [y, dx, dw, db],
                    [data1 * np.expand_dims(data2, -1) +
                     np.expand_dims(data3, -1),
                     grad1, grad2, grad3])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_affine_cuda(self):
        with dragon.device('cuda'):
            self.test_affine()

    def test_argmax(self):
        entries = [(0, True), (0, False), (1, True), (1, False)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, keep_dims in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    y = dragon.math.argmax(x, axis, keep_dims=keep_dims)
                    result = np.argmax(data, axis)
                    if keep_dims:
                        result = np.expand_dims(result, axis)
                    self.assertEqual(y, result)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_argmax_cuda(self):
        with dragon.device('cuda'):
            self.test_argmax()

    def test_argmin(self):
        entries = [(0, True), (0, False), (1, True), (1, False)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, keep_dims in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    y = dragon.math.argmin(x, axis, keep_dims=keep_dims)
                    result = np.argmin(data, axis)
                    if keep_dims:
                        result = np.expand_dims(result, axis)
                    self.assertEqual(y, result)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_argmin_cuda(self):
        with dragon.device('cuda'):
            self.test_argmin()

    def test_axpby(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1.], dtype='float32')
                x = new_tensor(data)
                dragon.math.axpby(x, x, alpha=2., beta=1.)
                self.assertEqual(x, data * 2. + data * 1.)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_axpby_cuda(self):
        with dragon.device('cuda'):
            self.test_axpby()

    def test_bitwise_and(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1 = np.random.binomial(1, 0.5, a_shape).astype('bool')
                    data2 = np.random.binomial(1, 0.5, b_shape).astype('bool')
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.bitwise.bitwise_and([a, b])
                    self.assertEqual(y, np.bitwise_and(data1, data2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_bitwise_and_cuda(self):
        with dragon.device('cuda'):
            self.test_bitwise_and()

    def test_bitwise_or(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1 = np.random.binomial(1, 0.5, a_shape).astype('bool')
                    data2 = np.random.binomial(1, 0.5, b_shape).astype('bool')
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.bitwise.bitwise_or([a, b])
                    self.assertEqual(y, np.bitwise_or(data1, data2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_bitwise_or_cuda(self):
        with dragon.device('cuda'):
            self.test_bitwise_or()

    def test_bitwise_xor(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1 = np.random.binomial(1, 0.5, a_shape).astype('bool')
                    data2 = np.random.binomial(1, 0.5, b_shape).astype('bool')
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.bitwise.bitwise_xor([a, b])
                    self.assertEqual(y, np.bitwise_xor(data1, data2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_bitwise_xor_cuda(self):
        with dragon.device('cuda'):
            self.test_bitwise_xor()

    def test_ceil(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([1.4, 1.7, 2.0])
                x = new_tensor(data)
                y = dragon.math.ceil(x)
                self.assertEqual(y, np.ceil(data))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_ceil_cuda(self):
        with dragon.device('cuda'):
            self.test_ceil()

    def test_clip(self):
        entries = [(None, None), (2, None), (None, 4), (2, 4)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for low, high in entries:
                    data, grad = arange((6,)), arange((6,))
                    if low is not None:
                        grad[grad < low] = 0
                    if high is not None:
                        grad[grad > high] = 0
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.math.clip(x, low, high)
                    dx = tape.gradient(y, [x], output_gradients=[x])[0]
                    result = np.clip(data, low, high) if low or high else data
                    self.assertEqual([y, dx], [result, grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_clip_cuda(self):
        with dragon.device('cuda'):
            self.test_clip()

    def test_cos(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([0., math.pi * 0.5, math.pi], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.cos(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [np.cos(data), data * (-np.sin(data))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_cos_cuda(self):
        with dragon.device('cuda'):
            self.test_cos()

    def test_cumsum(self):
        entries = [(0, False, False),
                   (0, True, False),
                   (0, False, True),
                   (0, True, True)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, exclusive, reverse in entries:
                    data, grad = arange((6,), 1), arange((6,), 1)
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.math.cumsum(x, axis, exclusive, reverse)
                    dx = tape.gradient(y, [x], output_gradients=[x])[0]
                    if reverse:
                        data = np.flipud(data)
                    else:
                        grad = np.flipud(grad)
                    if exclusive:
                        data = np.array([0] + data[:-1].tolist(), 'float32')
                        grad = np.array([0] + grad[:-1].tolist(), 'float32')
                    result = np.cumsum(data, axis)
                    grad = np.cumsum(grad, axis)
                    if reverse:
                        result = np.flipud(result)
                    else:
                        grad = np.flipud(grad)
                    self.assertEqual([y, dx], [result, grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_cumsum_cuda(self):
        with dragon.device('cuda'):
            self.test_cumsum()

    def test_div(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.math.div([a, b])
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    self.assertEqual(
                        [y, da, db],
                        [data1 / data2,
                         reduce_like(data3 / data2, data1),
                         reduce_like(data3 * (-data1) / np.square(data2), data2)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_div_cuda(self):
        with dragon.device('cuda'):
            self.test_div()

    def test_dot(self):
        entries = [((2,), (2,)), ((2, 3), (3, 2)), ((2, 3), (3,))]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in entries:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.math.dot([a, b])
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    if len(a_shape) == 1 and len(b_shape) == 1:
                        grad1, grad2 = data3 * data2, data3 * data1
                    elif len(a_shape) == 2 and len(b_shape) == 2:
                        grad1 = np.matmul(data3, data2.T)
                        grad2 = np.matmul(data1.T, data3)
                    elif len(a_shape) == 0 and len(b_shape) == 0:
                        grad1, grad2 = data2, data1
                    elif len(a_shape) >= 2 and len(b_shape) == 1:
                        grad1 = np.expand_dims(data3, -1) * data2
                        grad2 = np.dot(data1.T, data3)
                    else:
                        grad1 = grad2 = None, None
                    self.assertEqual([y, da, db], [np.dot(data1, data2), grad1, grad2])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_dot_cuda(self):
        with dragon.device('cuda'):
            self.test_dot()

    def test_equal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1 = uniform(a_shape)
                    data2 = dropout(data1, drop_ratio=0.5)
                    a, b = new_tensor(data1), new_tensor(data2)
                    y = dragon.math.equal([a, b])
                    self.assertEqual(y, np.equal(data1, data2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_equal_cuda(self):
        with dragon.device('cuda'):
            self.test_equal()

    def test_exp(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([0., 1., 2.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.exp(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [np.exp(data), data * np.exp(data)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_exp_cuda(self):
        with dragon.device('cuda'):
            self.test_exp()

    def test_floor(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([0.9, 1.4, 1.9])
                x = new_tensor(data)
                y = dragon.math.floor(x)
                self.assertEqual(y, np.floor(data))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_floor_cuda(self):
        with dragon.device('cuda'):
            self.test_floor()

    def test_fully_connected(self):
        entries = [((2, 3), (3, 4), (4,), False),
                   ((2, 3), (4, 3), (4,), True)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for x_shape, w_shape, b_shape, trans_w in entries:
                    data1, data2, data3 = arange(x_shape), arange(w_shape), arange(b_shape)
                    x, w, b = new_tensor(data1), new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w, b])
                        y = dragon.nn.fully_connected([x, w, b], transpose_w=trans_w)
                    data4 = arange(y.shape)
                    dy = new_tensor(data4)
                    dx, dw, db = tape.gradient(y, [x, w, b], output_gradients=[dy])
                    result = np.matmul(data1, data2.T if trans_w else data2) + data3
                    if trans_w:
                        grad1 = np.matmul(data4, data2)
                        grad2 = np.matmul(data4.T, data1)
                    else:
                        grad1 = np.matmul(data4, data2.T)
                        grad2 = np.matmul(data1.T, data4)
                    self.assertEqual(
                        [y, dx, dw, db],
                        [result, grad1, grad2, reduce_like(data4, data3)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_fully_connected_cuda(self):
        with dragon.device('cuda'):
            self.test_fully_connected()

    def test_greater(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    y = dragon.math.greater([a, b])
                    self.assertEqual(y, np.greater(data1, data2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_greater_cuda(self):
        with dragon.device('cuda'):
            self.test_greater()

    def test_greater_equal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    y = dragon.math.greater_equal([a, b])
                    self.assertEqual(y, np.greater_equal(data1, data2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_greater_equal_cuda(self):
        with dragon.device('cuda'):
            self.test_greater_equal()

    def test_is_inf(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([0., 1., float('inf')])
                x = new_tensor(data)
                y = dragon.math.is_inf(x)
                self.assertEqual(y, np.isinf(data))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_is_inf_cuda(self):
        with dragon.device('cuda'):
            self.test_is_inf()

    def test_is_nan(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([0., 1., float('nan')])
                x = new_tensor(data)
                y = dragon.math.is_nan(x)
                self.assertEqual(y, np.isnan(data))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_is_nan_cuda(self):
        with dragon.device('cuda'):
            self.test_is_nan()

    def test_less(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    y = dragon.math.less([a, b])
                    self.assertEqual(y, np.less(data1, data2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_less_cuda(self):
        with dragon.device('cuda'):
            self.test_less()

    def test_less_equal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    y = dragon.math.less_equal([a, b])
                    self.assertEqual(y, np.less_equal(data1, data2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_less_equal_cuda(self):
        with dragon.device('cuda'):
            self.test_less_equal()

    def test_log(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([1., 2., 3.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.log(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [np.log(data), (1. / data) * data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_log_cuda(self):
        with dragon.device('cuda'):
            self.test_log()

    def test_matmul(self):
        entries = [
            ((2, 3), (3, 4), False, False),
            ((2, 3), (4, 3), False, True),
            ((3, 2), (3, 4), True, False),
            ((3, 2), (4, 3), True, True)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape, trans_a, trans_b in entries:
                    data1, data2 = arange(a_shape), arange(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.math.matmul([a, b], trans_a, trans_b)
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    if trans_a:
                        if trans_b:
                            grad1 = np.matmul(data2.T, data3.T)
                            grad2 = np.matmul(data3.T, data1.T)
                        else:
                            grad1 = np.matmul(data2, data3.T)
                            grad2 = np.matmul(data1, data3)
                    else:
                        if trans_b:
                            grad1 = np.matmul(data3, data2)
                            grad2 = np.matmul(data3.T, data1)
                        else:
                            grad1 = np.matmul(data3, data2.T)
                            grad2 = np.matmul(data1.T, data3)
                    self.assertEqual(
                        [y, da, db],
                        [np.matmul(data1.T if trans_a else data1,
                                   data2.T if trans_b else data2), grad1, grad2])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_matmul_cuda(self):
        with dragon.device('cuda'):
            self.test_matmul()

    def test_max(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, keep_dims in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    y = dragon.math.max(x, axis, keep_dims=keep_dims)
                    result = np.max(data, axis, keepdims=keep_dims)
                    self.assertEqual(y, result)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_max_cuda(self):
        with dragon.device('cuda'):
            self.test_max()

    def test_maximum(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.math.maximum([a, b])
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    self.assertEqual(
                        [y, da, db],
                        [np.maximum(data1, data2),
                         reduce_like(data3 * (data1 > data2), data1),
                         reduce_like(data3 * (data1 <= data2), data2)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_maximum_cuda(self):
        with dragon.device('cuda'):
            self.test_maximum()

    def test_mean(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, keep_dims in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.math.mean(x, axis, keep_dims=keep_dims)
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    normalization = 1
                    for i in nest.flatten(axis):
                        normalization *= data.shape[i]
                    self.assertEqual(
                        [y, dx],
                        [np.mean(data, axis, keepdims=keep_dims),
                         broadcast_like(data3, data, axis) / normalization])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_mean_cuda(self):
        with dragon.device('cuda'):
            self.test_mean()

    def test_min(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, keep_dims in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    y = dragon.math.min(x, axis, keep_dims=keep_dims)
                    result = np.min(data, axis, keepdims=keep_dims)
                    self.assertEqual(y, result)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_min_cuda(self):
        with dragon.device('cuda'):
            self.test_min()

    def test_minimum(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.math.minimum([a, b])
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    self.assertEqual(
                        [y, da, db],
                        [np.minimum(data1, data2),
                         reduce_like(data3 * (data1 < data2), data1),
                         reduce_like(data3 * (data1 >= data2), data2)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_minimum_cuda(self):
        with dragon.device('cuda'):
            self.test_minimum()

    def test_moments(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, keep_dims in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    mean, var = dragon.math.moments(x, axis, keep_dims=keep_dims)
                    self.assertEqual(
                        [mean, var],
                        [np.array(np.mean(data, axis, keepdims=keep_dims)),
                         np.array(np.var(data, axis, keepdims=keep_dims))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_moments_cuda(self):
        with dragon.device('cuda'):
            self.test_moments()

    def test_mul(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 10)
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.math.mul([a, b])
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    self.assertEqual(
                        [y, da, db],
                        [data1 * data2,
                         reduce_like(data3 * data2, data1),
                         reduce_like(data3 * data1, data2)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_mul_cuda(self):
        with dragon.device('cuda'):
            self.test_mul()

    def test_negative(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.negative(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [np.negative(data), -data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_negative_cuda(self):
        with dragon.device('cuda'):
            self.test_negative()

    def test_not_equal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1 = uniform(a_shape)
                    data2 = dropout(data1, drop_ratio=0.5)
                    a, b = new_tensor(data1), new_tensor(data2)
                    y = dragon.math.not_equal([a, b])
                    self.assertEqual(y, np.not_equal(data1, data2))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_not_equal_cuda(self):
        with dragon.device('cuda'):
            self.test_not_equal()

    def test_pow(self, prec=None):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape, 1), arange(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.math.pow([a, b])
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    result = np.power(data1, data2)
                    self.assertEqual(
                        [y, da, db],
                        [np.power(data1, data2),
                         reduce_like(data3 * result * data2 / data1, data1),
                         reduce_like(data3 * result * np.log(data1), data2)], prec=prec)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_pow_cuda(self):
        with dragon.device('cuda'):
            self.test_pow(prec=1e-3)

    def test_reciprocal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([1., 2., 3.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.reciprocal(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = np.reciprocal(data)
                self.assertEqual([y, dx], [result, data * (-np.square(result))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_reciprocal_cuda(self):
        with dragon.device('cuda'):
            self.test_reciprocal()

    def test_round(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([0.9, 1.4, 1.9], 'float32')
                x = new_tensor(data)
                y = dragon.math.round(x)
                self.assertEqual(y, np.round(data))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_round_cuda(self):
        with dragon.device('cuda'):
            self.test_round()

    def test_rsqrt(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([4., 9., 16], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.rsqrt(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = 1. / np.sqrt(data)
                self.assertEqual([y, dx], [result, data * (-0.5 * (result * result * result))])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_rsqrt_cuda(self):
        with dragon.device('cuda'):
            self.test_rsqrt()

    def test_sign(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.sign(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual(
                    [y, dx],
                    [np.sign(data), np.zeros_like(data, 'float32')])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sign_cuda(self):
        with dragon.device('cuda'):
            self.test_sign()

    def test_sin(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([0., math.pi * 0.5, math.pi], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.sin(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [np.sin(data), data * np.cos(data)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sin_cuda(self):
        with dragon.device('cuda'):
            self.test_sin()

    def test_sqrt(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([4., 9., 16], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.sqrt(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                result = np.sqrt(data)
                self.assertEqual([y, dx], [result, data * 0.5 / result])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sqrt_cuda(self):
        with dragon.device('cuda'):
            self.test_sqrt()

    def test_square(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([2., 3., 4.], 'float32')
                x = new_tensor(data)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.math.square(x)
                dx = tape.gradient(y, [x], output_gradients=[x])[0]
                self.assertEqual([y, dx], [np.square(data), data * 2. * data])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_square_cuda(self):
        with dragon.device('cuda'):
            self.test_square()

    def test_sub(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([a, b])
                        y = dragon.math.sub([a, b])
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    da, db = tape.gradient(y, [a, b], output_gradients=[dy])
                    self.assertEqual(
                        [y, da, db],
                        [data1 - data2,
                         reduce_like(data3, data1),
                         reduce_like(-data3, data2)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sub_cuda(self):
        with dragon.device('cuda'):
            self.test_sub()

    def test_sum(self):
        entries = [(0, True), (0, False),
                   (1, True), (1, False),
                   ((0, 1), True), ((0, 1), False)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for axis, keep_dims in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.math.sum(x, axis, keep_dims=keep_dims)
                    data3 = arange(y.shape)
                    dy = new_tensor(data3)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual(
                        [y, dx],
                        [np.sum(data, axis, keepdims=keep_dims),
                         broadcast_like(data3, data, axis)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sum_cuda(self):
        with dragon.device('cuda'):
            self.test_sum()


class TestMetricOps(OpTestCase):
    """Test the metric ops."""

    def test_accuracy(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data1 = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], 'float32')
                data2 = np.array([0, 1], 'int64')
                a, b = new_tensor(data1), new_tensor(data2)
                y = dragon.metrics.accuracy([a, b])
                self.assertEqual(y, np.array(0.5))


class TestNormalizationOps(OpTestCase):
    """Test the normalization ops."""

    def __init__(self, method_name='runTest'):
        super(TestNormalizationOps, self).__init__(method_name)
        self.cudnn_ws = dragon.Workspace()

    def test_batch_norm(self):
        eps = 1e-5
        entries = [((4, 3), (3,), -1, 0),
                   ((4, 3), (3,), -1, 1),
                   ((4, 3, 2), (1, 3, 1), 1, 0),
                   ((4, 3, 2), (1, 3, 1), 1, 1),
                   ((4, 2, 3), (1, 1, 3), -1, 0),
                   ((4, 2, 3), (1, 1, 3), -1, 1)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for x_shape, w_shape, axis, use_stats in entries:
                    data1 = arange(x_shape) * .1
                    data2, data3 = arange(w_shape, 1) * .1, arange(w_shape) * .1
                    data4, data5 = arange(w_shape) * .1, arange(w_shape, 1) * .1
                    data6 = uniform(x_shape)
                    x = new_tensor(data1)
                    w, b = new_tensor(data2), new_tensor(data3)
                    rm, rv = new_tensor(data4), new_tensor(data5)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w, b])
                        y = dragon.nn.batch_norm(
                            [x, w, b, rm, rv],
                            axis=axis, use_stats=use_stats, eps=eps)
                    dy = new_tensor(data6)
                    dx, dw, db = tape.gradient(y, [x, w, b], output_gradients=[dy])
                    if use_stats == 0:
                        axes = list(range(0, len(data1.shape)))
                        axes.pop(axis)
                        mean = broadcast_like(np.mean(data1, tuple(axes)), data1, axes)
                        sig = broadcast_like(np.sqrt(np.var(data1, tuple(axes)) + eps), data1, axes)
                        result = (data1 - mean) / sig
                    else:
                        sig = np.sqrt(data5 + eps)
                        result = (data1 - data4) / sig
                    grad2 = reduce_like(data6 * result, data2)
                    grad3 = reduce_like(data6, data3)
                    if use_stats == 0:
                        scale = 1. / float(np.prod([data1.shape[i] for i in axes]))
                        grad1 = (data6 - (result * grad2 + grad3) * scale) * data2 / sig
                    else:
                        grad1 = data6 * data2 / sig
                    result = result * data2 + data3
                    self.assertEqual(
                        [y, dx, dw, db],
                        [result, grad1, grad2.flatten(), grad3.flatten()])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_batch_norm_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_batch_norm()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_batch_norm_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_batch_norm()

    def test_group_norm(self):
        eps = 1e-5
        entries = [((1, 4), (4,), -1, 2, (2,)),
                   ((1, 4, 2), (1, 4, 1), 1, 2, (2, 3)),
                   ((1, 2, 4), (1, 1, 4), -1, 2, (1, 3))]
        grads = [[[-0.0008, 0.0008, -0.00239, 0.00239]],
                 [[[0.03554, -0.06266], [0.01796, 0.00917],
                   [0.07091, -0.17008], [0.12537, -0.02621]]],
                 [[[0.00566, 0.0228, -0.02864, 0.07982],
                   [-0.1198, 0.09134, -0.17682, 0.12564]]]]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (x_shape, w_shape, axis, group, axes), grad1 in zip(entries, grads):
                    data1 = arange(x_shape) * .1
                    data2, data3 = arange(w_shape, 1) * .1, arange(w_shape) * .1
                    data6 = arange(x_shape) * .1
                    x = new_tensor(data1)
                    w, b = new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w, b])
                        y = dragon.nn.group_norm(
                            [x, w, b], axis=axis, group=group, eps=eps)
                    dy = new_tensor(data6)
                    dx, dw, db = tape.gradient(y, [x, w, b], output_gradients=[dy])
                    new_shape = list(x_shape[:])
                    new_shape[axis] //= group
                    new_shape.insert(axis, group)
                    data1 = data1.reshape(new_shape)
                    mean = broadcast_like(np.mean(data1, axes), data1, axes)
                    sig = broadcast_like(np.sqrt(np.var(data1, axes) + eps), data1, axes)
                    result = ((data1 - mean) / sig).reshape(x_shape)
                    grad2 = reduce_like(data6 * result, data2)
                    grad3 = reduce_like(data6, data3)
                    result = result * data2 + data3
                    self.assertEqual(
                        [y, dx, dw, db],
                        [result, np.array(grad1, data1.dtype),
                         grad2.flatten(), grad3.flatten()])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_group_norm_cuda(self):
        with dragon.device('cuda'):
            self.test_group_norm()

    def test_instance_norm(self):
        eps = 1e-5
        entries = [((1, 4), (4,), -1, 4, (2,)),
                   ((1, 4, 2), (1, 4, 1), 1, 4, (2, 3)),
                   ((1, 2, 4), (1, 1, 4), -1, 4, (1, 3))]
        grads = [[[0., 0., 0., 0.]],
                 [[[-0.03976, 0.03976], [-0.07951, 0.07951],
                   [-0.11921, 0.11919], [-0.15876, 0.15871]]],
                 [[[-0.0025, -0.005, -0.00749, -0.01],
                   [0.0025, 0.005, 0.00749, 0.01]]]]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (x_shape, w_shape, axis, group, axes), grad1 in zip(entries, grads):
                    data1 = arange(x_shape) * .1
                    data2, data3 = arange(w_shape, 1) * .1, arange(w_shape) * .1
                    data6 = arange(x_shape) * 10.
                    x = new_tensor(data1)
                    w, b = new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w, b])
                        y = dragon.nn.instance_norm([x, w, b], axis=axis, eps=eps)
                    dy = new_tensor(data6)
                    dx, dw, db = tape.gradient(y, [x, w, b], output_gradients=[dy])
                    new_shape = list(x_shape[:])
                    new_shape[axis] //= group
                    new_shape.insert(axis, group)
                    data1 = data1.reshape(new_shape)
                    mean = broadcast_like(np.mean(data1, axes), data1, axes)
                    sig = broadcast_like(np.sqrt(np.var(data1, axes) + eps), data1, axes)
                    result = ((data1 - mean) / sig).reshape(x_shape)
                    grad2 = reduce_like(data6 * result, data2)
                    grad3 = reduce_like(data6, data3)
                    result = result * data2 + data3
                    self.assertEqual(
                        [y, dw, db],
                        [result,
                         grad2.flatten(), grad3.flatten()], prec=1e-3)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_instance_norm_cuda(self):
        with dragon.device('cuda'):
            self.test_instance_norm()

    def test_layer_norm(self):
        eps = 1e-5
        entries = [((1, 4), (4,), -1, 1, (2,)),
                   ((1, 4, 2), (1, 4, 1), 1, 1, (2, 3)),
                   ((1, 2, 4), (1, 1, 4), -1, 1, (1, 3))]
        grads = [[[0.08898, -0.08955, -0.08926, 0.08984]],
                 [[[0.14534, 0.00719], [-0.04369, -0.13821],
                   [-0.05817, -0.10905], [0.10191, 0.09467]]],
                 [[[0.07264, 0.01448, 0.0436, 0.16],
                   [-0.33456, -0.21816, -0.01448, 0.27648]]]]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (x_shape, w_shape, axis, group, axes), grad1 in zip(entries, grads):
                    data1 = arange(x_shape) * .1
                    data2, data3 = arange(w_shape, 1) * .1, arange(w_shape) * .1
                    data6 = arange(x_shape) * 10.
                    x = new_tensor(data1)
                    w, b = new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w, b])
                        y = dragon.nn.layer_norm([x, w, b], axis=axis, eps=eps)
                    dy = new_tensor(data6)
                    dx, dw, db = tape.gradient(y, [x, w, b], output_gradients=[dy])
                    new_shape = list(x_shape[:])
                    new_shape[axis] //= group
                    new_shape.insert(axis, group)
                    data1 = data1.reshape(new_shape)
                    mean = broadcast_like(np.mean(data1, axes), data1, axes)
                    sig = broadcast_like(np.sqrt(np.var(data1, axes) + eps), data1, axes)
                    result = ((data1 - mean) / sig).reshape(x_shape)
                    grad2 = reduce_like(data6 * result, data2)
                    grad3 = reduce_like(data6, data3)
                    result = result * data2 + data3
                    self.assertEqual(
                        [y, dw, db],
                        [result,
                         grad2.flatten(), grad3.flatten()], prec=1e-4)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_layer_norm_cuda(self):
        with dragon.device('cuda'):
            self.test_layer_norm()

    def test_local_response_norm(self, test_cudnn=False, prec=None):
        entries = [((2, 3, 2, 2), 5, 0.0001, 0.75, 1., 'NCHW'),
                   ((2, 2, 2, 3), 5, 0.0001, 0.75, 1., 'NHWC')]
        results = [[[[[0., 0.1], [0.2, 0.29999]],
                     [[0.4, 0.49999], [0.59999, 0.69998]],
                     [[0.79999, 0.89999], [0.99998, 1.09997]]],
                    [[[1.19986, 1.29982], [1.39979, 1.49975]],
                     [[1.59981, 1.69977], [1.79973, 1.89968]],
                     [[1.99976, 2.09972], [2.19967, 2.29962]]]],
                   [[[[0., 0.1, 0.2], [0.3, 0.4, 0.5]],
                     [[0.59999, 0.69998, 0.79998], [0.89996, 0.99995, 1.09995]]],
                    [[[1.19991, 1.2999, 1.39989], [1.49983, 1.59982, 1.6998]],
                     [[1.79971, 1.89969, 1.99967], [2.09954, 2.19952, 2.2995]]]]]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (x_shape, size, alpha, beta, bias, data_format), \
                        result in zip(entries, results):
                    if not test_cudnn:
                        continue
                    data = arange(x_shape) * .1
                    x = new_tensor(data)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.nn.local_response_norm(
                            x,
                            size=size,
                            alpha=alpha,
                            beta=beta,
                            bias=bias,
                            data_format=data_format,
                        )
                    dx = tape.gradient(y, [x], output_gradients=[x])[0]
                    self.assertEqual([y, dx], [np.array(result), np.array(result)], prec=prec)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_local_response_norm_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_local_response_norm()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_local_response_norm_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_local_response_norm(test_cudnn=True, prec=1e-2)

    def test_lp_normalize(self):
        entries = [(0, 1, 1e-12, 'sum'),
                   (0, 1, 1e-12, 'mean'),
                   (0, 2, 1e-12, 'sum'),
                   (0, 2, 1e-12, 'mean')]
        grads = np.array(
            [[0.06667, -0.01778, -0.01111, -0.00444, 0.00222, 0.00889],
             [0.4, -0.10667, -0.06667, -0.02667, 0.01333, 0.05333],
             [0.13484, 0.09807, 0.06129, 0.02452, -0.01226, -0.04903],
             [0.33029, 0.24021, 0.15013, 0.06005, -0.03003, -0.12011]], 'float32')
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (axis, p, eps, reduction), grad in zip(entries, grads):
                    data1, data2 = arange((6,)), arange((6,)) / 10 + 1
                    x, dy = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.math.lp_normalize(x, axis, p, eps, reduction)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    norm = np.abs(data1) if p == 1 else np.square(data1)
                    norm = norm.sum() if reduction == 'sum' else norm.mean()
                    norm = norm if p == 1 else np.sqrt(norm)
                    self.assertEqual([y, dx], [data1 / max(norm, eps), grad])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_lp_normalize_cuda(self):
        with dragon.device('cuda'):
            self.test_lp_normalize()


class TestTensorOps(OpTestCase):

    # Testing shapes for binary ops
    unary_test_shapes = [(2,)]

    # Testing shapes for binary ops
    binary_test_shapes = [((2,), (2,)), ((2, 3), (3,)), ((2, 3), (2, 1))]

    def test_add(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a + b, data1 + data2)

    def test_astype(self):
        entries = [('int8', 'uint8'),
                   ('int32', 'float32'),
                   ('float32', 'int32'),
                   ('float32', 'float16'),
                   ('float32', 'float32'),
                   ('float32', 'float64')]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for in_type, out_type in entries:
                    data = np.array([-2., -1., 0., 1., 2.], dtype=in_type)
                    x = dragon.constant([-2., -1., 0., 1., 2.], dtype=in_type)
                    self.assertEqual(x.astype(out_type), data.astype(out_type))

    def test_copy(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = arange((4,))
                x = new_tensor(data)
                self.assertEqual(x.copy(), data)

    def test_constant(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                x = new_tensor(uniform((4,))).constant(1)
                if execution == 'EAGER_MODE':
                    self.assertEqual(x, np.ones((4,)))

    def test_div(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a / b, data1 / data2)

    def test_ge(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a >= b, data1 >= data2)

    def test_getitem(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = arange((2, 3))
                grad = np.zeros((2, 3), dtype='float32')
                grad[data > 2] = 1
                grad *= data
                x = new_tensor(data)
                self.assertEqual(x[x > 2], data[data > 2])
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
                        self.assertEqual(x.__getitem__(item), data.__getitem__(item))
                    except (NotImplementedError, ValueError, TypeError):
                        pass

    def test_glorot_normal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                new_tensor(arange((2, 3))).glorot_normal()

    def test_glorot_uniform(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                new_tensor(arange((2, 3))).glorot_uniform()

    def test_gt(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a > b, data1 > data2)

    def test_iadd(self):
        with execution_context().mode('EAGER_MODE'):
            for a_shape, b_shape in self.binary_test_shapes:
                data1, data2 = arange(a_shape), arange(b_shape, 1)
                a, b = new_tensor(data1), new_tensor(data2)
                a += b
                data1 += data2
                self.assertEqual(a, data1)

    def test_idiv(self):
        with execution_context().mode('EAGER_MODE'):
            for a_shape, b_shape in self.binary_test_shapes:
                data1, data2 = arange(a_shape), arange(b_shape, 1)
                a, b = new_tensor(data1), new_tensor(data2)
                a /= b
                data1 /= data2
                self.assertEqual(a, data1)

    def test_imul(self):
        with execution_context().mode('EAGER_MODE'):
            for a_shape, b_shape in self.binary_test_shapes:
                data1, data2 = arange(a_shape), arange(b_shape, 1)
                a, b = new_tensor(data1), new_tensor(data2)
                a *= b
                data1 *= data2
                self.assertEqual(a, data1)

    def test_isub(self):
        with execution_context().mode('EAGER_MODE'):
            for a_shape, b_shape in self.binary_test_shapes:
                data1, data2 = arange(a_shape), arange(b_shape, 1)
                a, b = new_tensor(data1), new_tensor(data2)
                a -= b
                data1 -= data2
                self.assertEqual(a, data1)

    def test_le(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a <= b, data1 <= data2)

    def test_lt(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = uniform(a_shape), uniform(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a < b, data1 < data2)

    def test_mul(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a * b, data1 * data2)

    def test_neg(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data = np.array([-1., 0., 1.], 'float32')
                x = new_tensor(data)
                self.assertEqual(-x, -data)

    def test_normal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                new_tensor(arange((2, 3))).normal()

    def test_radd(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a.__radd__(b), data2 + data1)

    def test_rdiv(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape, 1), arange(b_shape)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a.__rdiv__(b), data2 / data1)

    def test_reshape(self):
        entries = [(0, 0), (0, -1)]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for shape in entries:
                    data = arange((2, 3))
                    x = new_tensor(data)
                    y = x.reshape(shape)
                    self.assertEqual(y, data.reshape(y.shape))

    def test_rmul(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a.__rmul__(b), data2 * data1)

    def test_setitem(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
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

    def test_rsub(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a.__rsub__(b), data2 - data1)

    def test_sub(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for a_shape, b_shape in self.binary_test_shapes:
                    data1, data2 = arange(a_shape), arange(b_shape, 1)
                    a, b = new_tensor(data1), new_tensor(data2)
                    self.assertEqual(a - b, data1 - data2)

    def test_truncated_normal(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                new_tensor(arange((2, 3))).truncated_normal()

    def test_uniform(self):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                new_tensor(arange((2, 3))).uniform()


class TestTrainingOps(OpTestCase):
    """Test the training ops."""

    def __init__(self, method_name='runTest'):
        super(TestTrainingOps, self).__init__(method_name)
        self.adam = dragon.optimizers.Adam()
        self.nesterov = dragon.optimizers.Nesterov()
        self.rmsprop = dragon.optimizers.RMSprop()
        self.sgd = dragon.optimizers.SGD()

    def test_adam_update(self):
        with execution_context().mode('EAGER_MODE'):
            lr, eps = self.adam.base_lr, self.adam.eps
            beta1, beta2 = self.adam.beta1, self.adam.beta2
            data1 = uniform((2, 3))
            data2, data3 = np.zeros((2, 3), 'float32'), np.zeros((2, 3), 'float32')
            param = new_tensor(data1)
            for i in range(2):
                t = i + 1
                coef = math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                data4 = uniform((2, 3))
                grad = new_tensor(data4)
                self.adam._run_update(param, grad)
                data2 = beta1 * data2 + (1 - beta1) * data4
                data3 = beta2 * data3 + (1 - beta2) * np.square(data4)
                data1 -= (lr * coef * data2 / (np.sqrt(data3) + eps))
                self.assertEqual(param, data1)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_adam_update_cuda(self):
        with dragon.device('cuda'):
            self.test_adam_update()

    def test_nesterov_update(self):
        with execution_context().mode('EAGER_MODE'):
            momentum, lr = self.nesterov.momentum, self.nesterov.base_lr
            data1, data2 = uniform((2, 3)), np.zeros((2, 3), 'float32')
            param = new_tensor(data1)
            for i in range(2):
                data3 = uniform((2, 3))
                grad = new_tensor(data3)
                self.nesterov._run_update(param, grad)
                data2_new = momentum * data2 + lr * data3
                data1 -= (1 + momentum) * data2_new - momentum * data2
                data2 = data2_new
                self.assertEqual(param, data1)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_nesterov_update_cuda(self):
        with dragon.device('cuda'):
            self.test_nesterov_update()

    def test_rmsprop_update(self):
        with execution_context().mode('EAGER_MODE'):
            momentum, lr = self.rmsprop.momentum, self.rmsprop.base_lr
            decay, eps = self.rmsprop.decay, self.rmsprop.eps
            data1 = uniform((2, 3))
            data2, data3 = np.zeros((2, 3), 'float32'), np.zeros((2, 3), 'float32')
            param = new_tensor(data1)
            for i in range(2):
                data4 = uniform((2, 3))
                grad = new_tensor(data4)
                self.rmsprop._run_update(param, grad)
                data3 = decay * data3 + (1 - decay) * np.square(data4)
                data2 = momentum * data2 + (lr * data4 / (np.sqrt(data3) + eps))
                data1 -= data2
                self.assertEqual(param, data1)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_rmsprop_update_cuda(self):
        with dragon.device('cuda'):
            self.test_rmsprop_update()

    def test_sgd_update(self):
        with execution_context().mode('EAGER_MODE'):
            momentum, lr = self.sgd.momentum, self.sgd.base_lr
            data1, data2 = uniform((2, 3)), np.zeros((2, 3), 'float32')
            param = new_tensor(data1)
            for i in range(2):
                data3 = uniform((2, 3))
                grad = new_tensor(data3)
                self.sgd._run_update(param, grad)
                data2 = momentum * data2 + lr * data3
                data1 -= data2
                self.assertEqual(param, data1)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_sgd_update_cuda(self):
        with dragon.device('cuda'):
            self.test_sgd_update()


class TestVisionOps(OpTestCase):
    """Test the vision ops."""

    def __init__(self, method_name='runTest'):
        super(TestVisionOps, self).__init__(method_name)
        self.cudnn_ws = dragon.Workspace()

    def test_bias_add(self):
        entries = [((2, 3), (3,), 'NCHW'),
                   ((2, 3, 4), (1, 3, 1), 'NCHW'),
                   ((2, 4, 3), (1, 1, 3), 'NHWC')]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for x_shape, b_shape, data_format in entries:
                    data1, data2 = arange(x_shape), arange(b_shape)
                    x, w = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w])
                        y = dragon.nn.bias_add([x, w], data_format)
                    dx, dw = tape.gradient(y, [x, w], output_gradients=[x])
                    self.assertEqual(
                        [y, dx, dw],
                        [data1 + data2, data1, reduce_like(data1, data2).flatten()])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_bias_add_cuda(self):
        with dragon.device('cuda'):
            self.test_bias_add()

    def test_conv2d(self, prec=None):
        entries = [((2, 2, 2, 2), (3, 2, 1, 1), (3,), 1, 1, 0, 1, 1, 'NCHW'),
                   ((2, 2, 2, 2), (3, 2, 3, 3), (3,), 3, 1, 1, 1, 1, 'NCHW'),
                   ((2, 2, 2, 2), (3, 2, 1, 1), (3,), 1, 1, 0, 1, 1, 'NHWC'),
                   ((2, 2, 2, 2), (3, 2, 3, 3), (3,), 3, 1, 1, 1, 1, 'NHWC')]
        results = [[[[[0.04, 0.05], [0.06, 0.07]], [[0.22, 0.27], [0.32, 0.37]], [[0.4, 0.49], [0.58, 0.67]]],
                    [[[0.12, 0.13], [0.14, 0.15]], [[0.62, 0.67], [0.72, 0.77]], [[1.12, 1.21], [1.3, 1.39]]]],
                   [[[[3.8, 3.52], [2.96, 2.68]], [[8.94, 8.66], [8.1, 7.82]], [[14.08, 13.8], [13.24, 12.96]]],
                    [[[10.52, 9.6], [7.76, 6.84]], [[27.18, 26.26], [24.42, 23.5]], [[43.84, 42.92], [41.08, 40.16]]]],
                   [[[[0.01, 0.13, 0.25], [0.03, 0.23, 0.43]], [[0.05, 0.33, 0.61], [0.07, 0.43, 0.79]]],
                    [[[0.09, 0.53, 0.97], [0.11, 0.63, 1.15]], [[0.13, 0.73, 1.33], [0.15, 0.83, 1.51]]]],
                   [[[[4.08, 9.22, 14.36], [3.52, 8.66, 13.8]], [[2.4, 7.54, 12.68], [1.84, 6.98, 12.12]]],
                    [[[12.08, 28.74, 45.4], [10.24, 26.9, 43.56]], [[6.56, 23.22, 39.88], [4.72, 21.38, 38.04]]]]]
        grads1 = [[[[[0.4, 0.46], [0.52, 0.58]], [[0.52, 0.61], [0.7, 0.79]]],
                   [[[1.12, 1.18], [1.24, 1.3]], [[1.6, 1.69], [1.78, 1.87]]]],
                  [[[[18.75, 19.41], [20.73, 21.39]], [[24.69, 25.35], [26.67, 27.33]]],
                   [[[47.55, 49.65], [53.85, 55.95]], [[66.45, 68.55], [72.75, 74.85]]]],
                  [[[[0.1, 0.13], [0.28, 0.4]], [[0.46, 0.67], [0.64, 0.94]]],
                   [[[0.82, 1.21], [1., 1.48]], [[1.18, 1.75], [1.36, 2.02]]]],
                  [[[[14.7, 15.36], [16.02, 16.68]], [[18.66, 19.32], [19.98, 20.64]]],
                   [[[46.38, 48.48], [50.58, 52.68]], [[58.98, 61.08], [63.18, 65.28]]]]]
        grads2 = [[[[[5.32]], [[7.72]]], [[[7.08]], [[10.76]]], [[[8.84]], [[13.8]]]],
                  [[[[1.2, 2.5, 1.28], [2.6, 5.32, 2.68], [1.32, 2.66, 1.32]],
                    [[1.92, 3.86, 1.92], [3.88, 7.72, 3.8], [1.88, 3.7, 1.8]]],
                   [[[1.52, 3.22, 1.68], [3.4, 7.08, 3.64], [1.8, 3.7, 1.88]],
                    [[2.56, 5.22, 2.64], [5.32, 10.76, 5.4], [2.68, 5.38, 2.68]]],
                   [[[1.84, 3.94, 2.08], [4.2, 8.84, 4.6], [2.28, 4.74, 2.44]],
                    [[3.2, 6.58, 3.36], [6.76, 13.8, 7.], [3.48, 7.06, 3.56]]]],
                  [[[[8.4, 9.24]]], [[[8.96, 9.88]]], [[[9.52, 10.52]]]],
                  [[[[1.68, 1.98], [3.72, 4.26], [1.92, 2.16]],
                    [[4.08, 4.56], [8.4, 9.24], [4.08, 4.44]],
                    [[1.92, 2.1], [3.72, 4.02], [1.68, 1.8]]],
                   [[[1.76, 2.08], [3.92, 4.5], [2.04, 2.3]],
                    [[4.32, 4.84], [8.96, 9.88], [4.4, 4.8]],
                    [[2.08, 2.28], [4.08, 4.42], [1.88, 2.02]]],
                   [[[1.84, 2.18], [4.12, 4.74], [2.16, 2.44]],
                    [[4.56, 5.12], [9.52, 10.52], [4.72, 5.16]],
                    [[2.24, 2.46], [4.44, 4.82], [2.08, 2.24]]]]]
        grads3 = [[6., 9.2, 12.4], [6., 9.2, 12.4], [8.4, 9.2, 10.], [8.4, 9.2, 10.]]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (x_shape, w_shape, b_shape, kernel_shape,
                        strides, pads, dilations, group, data_format), \
                        result, grad1, grad2, grad3 in zip(entries, results, grads1, grads2, grads3):
                    data1, data2, data3 = arange(x_shape) * .1, arange(w_shape) * .1, arange(b_shape) * .1
                    x, w, b = new_tensor(data1), new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w, b])
                        y = dragon.nn.conv2d(
                            [x, w, b],
                            kernel_shape=kernel_shape,
                            strides=strides,
                            pads=pads,
                            dilations=dilations,
                            group=group,
                            data_format=data_format,
                        )
                    data4 = arange(y.shape) * .1
                    dy = new_tensor(data4)
                    dx, dw, db = tape.gradient(y, [x, w, b], output_gradients=[dy])
                    self.assertEqual(
                        [y, dx, dw],
                        [np.array(result),
                         np.array(grad1),
                         np.array(grad2).reshape(w_shape),
                         np.array(grad3)], prec=prec)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_conv2d_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_conv2d()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_conv2d_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_conv2d(prec=1e-4)

    def test_conv2d_transpose(self, prec=None):
        entries = [((2, 2, 2, 2), (2, 3, 1, 1), (3,), 1, 1, 0, 1, 1, 'NCHW'),
                   ((2, 2, 2, 2), (2, 3, 3, 3), (3,), 3, 1, 1, 1, 1, 'NCHW'),
                   ((2, 2, 2, 2), (2, 3, 1, 1), (3,), 1, 1, 0, 1, 1, 'NHWC'),
                   ((2, 2, 2, 2), (2, 3, 3, 3), (3,), 3, 1, 1, 1, 1, 'NHWC')]
        results = [[[[[0.12, 0.15], [0.18, 0.21]], [[0.26, 0.31], [0.36, 0.41]], [[0.4, 0.47], [0.54, 0.61]]],
                    [[[0.36, 0.39], [0.42, 0.45]], [[0.66, 0.71], [0.76, 0.81]], [[0.96, 1.03], [1.1, 1.17]]]],
                   [[[[6.36, 6.64], [7.2, 7.48]],
                     [[8.98, 9.26], [9.82, 10.1]],
                     [[11.6, 11.88], [12.44, 12.72]]],
                    [[[16.28, 17.2], [19.04, 19.96]],
                     [[24.66, 25.58], [27.42, 28.34]],
                     [[33.04, 33.96], [35.8, 36.72]]]],
                   [[[[0.03, 0.14, 0.25], [0.09, 0.24, 0.39]], [[0.15, 0.34, 0.53], [0.21, 0.44, 0.67]]],
                    [[[0.27, 0.54, 0.81], [0.33, 0.64, 0.95]], [[0.39, 0.74, 1.09], [0.45, 0.84, 1.23]]]],
                   [[[[5.16, 5.54, 5.92], [6., 6.38, 6.76]], [[7.68, 8.06, 8.44], [8.52, 8.9, 9.28]]],
                    [[[17.64, 18.66, 19.68], [20.4, 21.42, 22.44]], [[25.92, 26.94, 27.96], [28.68, 29.7, 30.72]]]]]
        grads1 = [[[[[0.2, 0.23], [0.26, 0.29]], [[0.56, 0.68], [0.8, 0.92]]],
                  [[[0.56, 0.59], [0.62, 0.65]], [[2., 2.12], [2.24, 2.36]]]],
                  [[[[12.99, 12.33], [11.01, 10.35]], [[30.81, 30.15], [28.83, 28.17]]],
                   [[[34.59, 32.49], [28.29, 26.19]], [[91.29001, 89.19], [84.99, 82.89]]]],
                  [[[[0.05, 0.14], [0.14, 0.5]], [[0.23, 0.86], [0.32, 1.22]]],
                   [[[0.41, 1.58], [0.5, 1.94]], [[0.59, 2.3], [0.68, 2.66]]]],
                  [[[[14.51, 32.33], [12.53, 30.35]], [[8.57, 26.39], [6.59, 24.41]]],
                   [[[41.87, 98.57], [35.57, 92.27]], [[22.97, 79.67], [16.67, 73.37]]]]]
        grads2 = [[[[[5.32]], [[7.08]], [[8.84]]], [[[7.72]], [[10.76]], [[13.8]]]],
                  [[[[1.32, 2.66, 1.32], [2.68, 5.32, 2.6], [1.28, 2.5, 1.2]],
                    [[1.88, 3.7, 1.8], [3.64, 7.08, 3.4], [1.68, 3.22, 1.52]],
                    [[2.44, 4.74, 2.28], [4.6, 8.84, 4.2], [2.08, 3.94, 1.84]]],
                   [[[1.8, 3.7, 1.88], [3.8, 7.72, 3.88], [1.92, 3.86, 1.92]],
                    [[2.68, 5.38, 2.68], [5.4, 10.76, 5.32], [2.64, 5.22, 2.56]],
                    [[3.56, 7.06, 3.48], [7., 13.8, 6.76], [3.36, 6.58, 3.2]]]],
                  [[[[8.4, 8.96, 9.52]]], [[[9.24, 9.88, 10.52]]]],
                  [[[[1.68, 1.88, 2.08], [3.72, 4.08, 4.44], [1.92, 2.08, 2.24]],
                    [[4.08, 4.4, 4.72], [8.4, 8.96, 9.52], [4.08, 4.32, 4.56]],
                    [[1.92, 2.04, 2.16], [3.72, 3.92, 4.12], [1.68, 1.76, 1.84]]],
                   [[[1.8, 2.02, 2.24], [4.02, 4.42, 4.82], [2.1, 2.28, 2.46]],
                    [[4.44, 4.8, 5.16], [9.24, 9.88, 10.52], [4.56, 4.84, 5.12]],
                    [[2.16, 2.3, 2.44], [4.26, 4.5, 4.74], [1.98, 2.08, 2.18]]]]]
        grads3 = [[6., 9.2, 12.4], [6., 9.2, 12.4], [8.4, 9.2, 10.], [8.4, 9.2, 10.]]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (x_shape, w_shape, b_shape, kernel_shape,
                        strides, pads, dilations, group, data_format),\
                        result, grad1, grad2, grad3 in zip(entries, results, grads1, grads2, grads3):
                    data1, data2, data3 = arange(x_shape) * .1, arange(w_shape) * .1, arange(b_shape) * .1
                    x, w, b = new_tensor(data1), new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w, b])
                        y = dragon.nn.conv2d_transpose(
                            [x, w, b],
                            kernel_shape=kernel_shape,
                            strides=strides,
                            pads=pads,
                            dilations=dilations,
                            group=group,
                            data_format=data_format,
                        )
                    data4 = arange(y.shape) * .1
                    dy = new_tensor(data4)
                    dx, dw, db = tape.gradient(y, [x, w, b], output_gradients=[dy])
                    self.assertEqual(
                        [y, dx, dw, db],
                        [np.array(result),
                         np.array(grad1),
                         np.array(grad2).reshape(w_shape),
                         np.array(grad3)], prec=prec)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_conv2d_transpose_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_conv2d_transpose()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_conv2d_transpose_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_conv2d_transpose(prec=1e-4)

    def test_depthwise_conv2d(self, test_grad=False):
        entries = [((2, 2, 2, 2), (2, 1, 1, 1), (2,), 1, 1, 0, 1, 'NCHW'),
                   ((2, 2, 2, 2), (2, 1, 3, 3), (2,), 3, 1, 1, 1, 'NCHW'),
                   ((2, 2, 2, 2), (2, 1, 1, 1), (2,), 1, 1, 0, 1, 'NHWC'),
                   ((2, 2, 2, 2), (2, 1, 3, 3), (2,), 3, 1, 1, 1, 'NHWC')]
        results = [[[[[0., 0.], [0., 0.]], [[0.14, 0.15], [0.16, 0.17]]],
                    [[[0., 0.], [0., 0.]], [[0.22, 0.23], [0.24, 0.25]]]],
                   [[[[0.43, 0.37], [0.25, 0.19]], [[3.47, 3.25], [2.81, 2.59]]],
                    [[[2.35, 1.97], [1.21, 0.83]], [[8.27, 7.73], [6.65, 6.11]]]],
                   [[[[0., 0.11], [0., 0.13]], [[0., 0.15], [0., 0.17]]],
                    [[[0., 0.19], [0., 0.21]], [[0., 0.23], [0., 0.25]]]],
                   [[[[0.86, 2.64], [0.74, 2.48]], [[0.5, 2.16], [0.38, 2.]]],
                    [[[2.78, 7.44], [2.34, 6.96]], [[1.46, 6.], [1.02, 5.52]]]]]
        grads1 = [[[[[0., 0.], [0., 0.]], [[0.04, 0.05], [0.06, 0.07]]],
                   [[[0., 0.], [0., 0.]], [[0.12, 0.13], [0.14, 0.15]]]],
                  [[[[0.05, 0.11], [0.23, 0.29]], [[2.35, 2.57], [3.01, 3.23]]],
                   [[[0.69, 1.07], [1.83, 2.21]], [[5.87, 6.41], [7.49, 8.03]]]],
                  [[[[0., 0.01], [0., 0.03]], [[0., 0.05], [0., 0.07]]],
                   [[[0., 0.09], [0., 0.11]], [[0., 0.13], [0., 0.15]]]],
                  [[[[0.1, 1.62], [0.22, 1.78]], [[0.46, 2.1], [0.58, 2.26]]],
                   [[[0.74, 5.14], [1.18, 5.62]], [[2.06, 6.58], [2.5, 7.06]]]]]
        grads2 = [[[[[3.8]]], [[[8.6]]]],
                  [[[[0.88, 1.82, 0.92], [1.88, 3.8, 1.88], [0.92, 1.82, 0.88]]],
                   [[[2.08, 4.22, 2.12], [4.28, 8.6, 4.28], [2.12, 4.22, 2.08]]]],
                  [[[[5.6]]], [[[6.8]]]],
                  [[[[1.12], [2.48], [1.28]], [[2.72], [5.6], [2.72]], [[1.28], [2.48], [1.12]]],
                   [[[1.42], [3.08], [1.58]], [[3.32], [6.8], [3.32]], [[1.58], [3.08], [1.42]]]]]
        grads3 = [[4.4, 7.6], [4.4, 7.6], [5.6, 6.4], [5.6, 6.4]]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (x_shape, w_shape, b_shape, kernel_shape,
                        strides, pads, dilations, data_format), \
                        result, grad1, grad2, grad3 in zip(entries, results, grads1, grads2, grads3):
                    data1, data2, data3 = arange(x_shape) * .1, arange(w_shape) * .1, arange(b_shape) * .1
                    x, w, b = new_tensor(data1), new_tensor(data2), new_tensor(data3)
                    with dragon.GradientTape() as tape:
                        tape.watch([x, w, b])
                        y = dragon.nn.depthwise_conv2d(
                            [x, w, b],
                            kernel_shape=kernel_shape,
                            strides=strides,
                            pads=pads,
                            dilations=dilations,
                            data_format=data_format,
                        )
                    if test_grad:
                        data4 = arange(y.shape) * .1
                        dy = new_tensor(data4)
                        dx, dw, db = tape.gradient(y, [x, w, b], output_gradients=[dy])
                        self.assertEqual(
                            [y, dx, dw, db],
                            [np.array(result),
                             np.array(grad1),
                             np.array(grad2).reshape(w_shape),
                             np.array(grad3)])
                    else:
                        self.assertEqual(y, np.array(result))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_depthwise_conv2d_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_depthwise_conv2d(test_grad=True)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_depthwise_conv2d_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_depthwise_conv2d(test_grad=True)

    def test_depth_to_space(self):
        n, co, si = 2, 2, 2
        entries = [(2, 2, 'NCHW'), (2, 3, 'NCHW'), (2, 2, 'NHWC'), (2, 3, 'NHWC')]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
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
                    x, dy = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.nn.depth_to_space(
                            x,
                            block_size=bs,
                            data_format=data_format)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [data2, data1])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_depth_to_space_cuda(self):
        with dragon.device('cuda'):
            self.test_depth_to_space()

    def test_pool2d(self):
        entries = [((2, 2, 2, 2), (2, 2), 2, 1, 'MAX', 'NCHW'),
                   ((2, 2, 2, 2), (2, 2), 2, 1, 'AVG', 'NCHW'),
                   ((2, 2, 2, 2), (2, 2), 2, 1, 'MAX', 'NHWC'),
                   ((2, 2, 2, 2), (2, 2), 2, 1, 'AVG', 'NHWC')]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for x_shape, kernel_shape, strides, pads, mode, data_format in entries:
                    data1 = arange(x_shape) * .1
                    x = new_tensor(data1)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.nn.pool2d(
                            x,
                            kernel_shape=kernel_shape,
                            strides=strides,
                            pads=pads,
                            mode=mode,
                            data_format=data_format,
                        )
                    data2 = arange(y.shape) * .1
                    dy = new_tensor(data2)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    result = data1 / (np.prod(kernel_shape) if mode == 'AVG' else 1.)
                    self.assertEqual([y, dx], [result, result])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_pool2d_cuda(self):
        dragon.cuda.enable_cudnn(False)
        with dragon.device('cuda'):
            self.test_pool2d()

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_pool2d_cudnn(self):
        dragon.cuda.enable_cudnn(True)
        with dragon.device('cuda'), self.cudnn_ws.as_default():
            self.test_pool2d()

    def test_resize(self):
        entries = [((2, 2, 1, 1), (2, 2), 'nearest', 'NCHW'),
                   ((2, 2, 1, 1), (2, 2), 'linear', 'NCHW'),
                   ((2, 2, 4, 4), (2, 2), 'nearest', 'NCHW'),
                   ((2, 2, 4, 4), (2, 2), 'linear', 'NCHW'),
                   ((2, 1, 1, 2), (2, 2), 'nearest', 'NHWC'),
                   ((2, 1, 1, 2), (2, 2), 'linear', 'NHWC'),
                   ((2, 4, 4, 2), (2, 2), 'nearest', 'NHWC'),
                   ((2, 4, 4, 2), (2, 2), 'linear', 'NHWC')]
        results = [[[[[0., 0.], [0., 0.]], [[0.1, 0.1], [0.1, 0.1]]],
                    [[[0.2, 0.2], [0.2, 0.2]], [[0.3, 0.3], [0.3, 0.3]]]],
                   [[[[0., 0.], [0., 0.]], [[0.1, 0.1], [0.1, 0.1]]],
                    [[[0.2, 0.2], [0.2, 0.2]], [[0.3, 0.3], [0.3, 0.3]]]],
                   [[[[0., 0.2], [0.8, 1.]], [[1.6, 1.8], [2.4, 2.6]]],
                    [[[3.2, 3.4], [4., 4.2]], [[4.8, 5.], [5.6, 5.8]]]],
                   [[[[0.25, 0.45], [1.05, 1.25]], [[1.85, 2.05], [2.65, 2.85]]],
                    [[[3.45, 3.65], [4.25, 4.45]], [[5.05, 5.25], [5.85, 6.05]]]],
                   [[[[0., 0.1], [0., 0.1]], [[0., 0.1], [0., 0.1]]],
                    [[[0.2, 0.3], [0.2, 0.3]], [[0.2, 0.3], [0.2, 0.3]]]],
                   [[[[0., 0.1], [0., 0.1]], [[0., 0.1], [0., 0.1]]],
                    [[[0.2, 0.3], [0.2, 0.3]], [[0.2, 0.3], [0.2, 0.3]]]],
                   [[[[0., 0.1], [0.4, 0.5]], [[1.6, 1.7], [2., 2.1]]],
                    [[[3.2, 3.3], [3.6, 3.7]], [[4.8, 4.9], [5.2, 5.3]]]],
                   [[[[0.5, 0.6], [0.9, 1.]], [[2.1, 2.2], [2.5, 2.6]]],
                    [[[3.7, 3.8], [4.1, 4.2]], [[5.3, 5.4], [5.7, 5.8]]]]]
        grads = [[[[[0.6]], [[2.2]]], [[[3.8]], [[5.4]]]],
                 [[[[0.6]], [[2.2]]], [[[3.8]], [[5.4]]]],
                 [[[[0., 0., 0.1, 0.], [0., 0., 0., 0.], [0.2, 0., 0.3, 0.], [0., 0., 0., 0.]],
                   [[0.4, 0., 0.5, 0.], [0., 0., 0., 0.], [0.6, 0., 0.7, 0.], [0., 0., 0., 0.]]],
                  [[[0.8, 0., 0.9, 0.], [0., 0., 0., 0.], [1., 0., 1.1, 0.], [0., 0., 0., 0.]],
                   [[1.2, 0., 1.3, 0.], [0., 0., 0., 0.], [1.4, 0., 1.5, 0.], [0., 0., 0., 0.]]]],
                 [[[[0., 0., 0.025, 0.025], [0., 0., 0.025, 0.025],
                    [0.05, 0.05, 0.075, 0.075], [0.05, 0.05, 0.075, 0.075]],
                   [[0.1, 0.1, 0.125, 0.125], [0.1, 0.1, 0.125, 0.125],
                    [0.15, 0.15, 0.175, 0.175], [0.15, 0.15, 0.175, 0.175]]],
                  [[[0.2, 0.2, 0.225, 0.225], [0.2, 0.2, 0.225, 0.225],
                    [0.25, 0.25, 0.275, 0.275], [0.25, 0.25, 0.275, 0.275]],
                   [[0.3, 0.3, 0.325, 0.325], [0.3, 0.3, 0.325, 0.325],
                    [0.35, 0.35, 0.375, 0.375], [0.35, 0.35, 0.375, 0.375]]]],
                 [[[[1.2, 1.6]]], [[[4.4, 4.8]]]],
                 [[[[1.2, 1.6]]], [[[4.4, 4.8]]]],
                 [[[[0., 0.1], [0., 0.], [0.2, 0.3], [0., 0.]],
                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                   [[0.4, 0.5], [0., 0.], [0.6, 0.7], [0., 0.]],
                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]],
                  [[[0.8, 0.9], [0., 0.], [1., 1.1], [0., 0.]],
                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                   [[1.2, 1.3], [0., 0.], [1.4, 1.5], [0., 0.]],
                   [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]],
                 [[[[0., 0.025], [0., 0.025], [0.05, 0.075], [0.05, 0.075]],
                   [[0., 0.025], [0., 0.025], [0.05, 0.075], [0.05, 0.075]],
                   [[0.1, 0.125], [0.1, 0.125], [0.15, 0.175], [0.15, 0.175]],
                   [[0.1, 0.125], [0.1, 0.125], [0.15, 0.175], [0.15, 0.175]]],
                  [[[0.2, 0.225], [0.2, 0.225], [0.25, 0.275], [0.25, 0.275]],
                   [[0.2, 0.225], [0.2, 0.225], [0.25, 0.275], [0.25, 0.275]],
                   [[0.3, 0.325], [0.3, 0.325], [0.35, 0.375], [0.35, 0.375]],
                   [[0.3, 0.325], [0.3, 0.325], [0.35, 0.375], [0.35, 0.375]]]]]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                for (x_shape, sizes, mode, data_format), result, grad \
                        in zip(entries, results, grads):
                    data1 = arange(x_shape) * .1
                    x = new_tensor(data1)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.vision.resize(
                            x,
                            sizes=sizes,
                            mode=mode,
                            data_format=data_format,
                            align_corners=False,
                        )
                    data2 = arange(y.shape) * .1
                    dy = new_tensor(data2)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [np.array(result), np.array(grad)])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_resize_cuda(self):
        with dragon.device('cuda'):
            self.test_resize()

    def test_roi_align(self, test_grad=False):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data1 = arange((2, 2, 2, 2))
                data2 = np.array([[0., 0., 0., 1., 1.],
                                  [1., 0., 0., 1., 1.]], 'float32')
                result = np.array([[[[1.5]], [[5.5]]], [[[9.5]], [[13.5]]]])
                grad = np.array([[[[0.025, 0.025], [0.025, 0.025]], [[0.05, 0.05], [0.05, 0.05]]],
                                 [[[0.075, 0.075], [0.075, 0.075]], [[0.1, 0.1], [0.1, 0.1]]]])
                x, roi = new_tensor(data1), new_tensor(data2)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.vision.roi_align(
                        [x, roi],
                        pooled_h=1,
                        pooled_w=1,
                        spatial_scale=1.,
                    )
                if test_grad:
                    data3 = arange(y.shape, 1) * .1
                    dy = new_tensor(data3)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [result, grad])
                else:
                    self.assertEqual(y, result)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_roi_align_cuda(self):
        with dragon.device('cuda'):
            self.test_roi_align(test_grad=True)

    def test_roi_pool(self, test_grad=False):
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
                data1 = arange((2, 2, 2, 2))
                data2 = np.array([[0., 0., 0., 1., 1.],
                                  [1., 0., 0., 1., 1.]], 'float32')
                grad = np.array([[[[0., 0.], [0., 0.1]], [[0., 0.], [0., 0.2]]],
                                 [[[0., 0.], [0., 0.3]], [[0., 0.], [0., 0.4]]]])
                x, roi = new_tensor(data1), new_tensor(data2)
                with dragon.GradientTape() as tape:
                    tape.watch(x)
                    y = dragon.vision.roi_pool(
                        [x, roi],
                        pooled_h=1,
                        pooled_w=1,
                        spatial_scale=1.,
                    )
                if test_grad:
                    data3 = arange(y.shape, 1) * .1
                    dy = new_tensor(data3)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [data1.max((2, 3), keepdims=True), grad])
                else:
                    self.assertEqual(y, data1.max((2, 3), keepdims=True))

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_roi_pool_cuda(self):
        with dragon.device('cuda'):
            self.test_roi_pool(test_grad=True)

    def test_space_to_depth(self):
        n, ci, so = 2, 2, 2
        entries = [(2, 2, 'NCHW'), (2, 3, 'NCHW'), (2, 2, 'NHWC'), (2, 3, 'NHWC')]
        for execution in ('EAGER_MODE', 'GRAPH_MODE'):
            with execution_context().mode(execution):
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
                    x, dy = new_tensor(data1), new_tensor(data2)
                    with dragon.GradientTape() as tape:
                        tape.watch(x)
                        y = dragon.nn.space_to_depth(
                            x,
                            block_size=bs,
                            data_format=data_format)
                    dx = tape.gradient(y, [x], output_gradients=[dy])[0]
                    self.assertEqual([y, dx], [data2, data1])

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_space_to_depth_cuda(self):
        with dragon.device('cuda'):
            self.test_space_to_depth()


def arange(shape, start=0, dtype='float32'):
    """Return the arange data with given shape."""
    return np.arange(start, start + int(np.prod(shape)), dtype=dtype).reshape(shape)


def broadcast_like(data, other, axes):
    """Broadcast data like the other."""
    shape = list(other.shape[:])
    for i in nest.flatten(axes):
        shape[i] = 1
    return data.reshape(shape) * np.ones_like(other, data.dtype)


def dropout(data, drop_ratio=0.5):
    """Return the random dropped data."""
    return data * np.random.binomial(1, 1. - drop_ratio, data.shape).astype(data.dtype)


def new_tensor(data):
    """Create a new tensor for current execution."""
    if execution_context().executing_eagerly():
        return dragon.EagerTensor(data, copy=True)
    return dragon.Tensor(None, data.shape, str(data.dtype)).set_value(data)


def process_indices(item):
    """Process and normalize the indices."""
    if not isinstance(item, (slice, tuple)):
        if not isinstance(item, int):
            raise ValueError('The index should be a integer.')
        item = (item,)
    if not isinstance(item, tuple):
        item = tuple([item])
    starts, sizes = [], []
    for ix, it in enumerate(item):
        if isinstance(it, slice):
            if it.start is None:
                starts.append(0)
            else:
                starts.append(it.start)
            if it.stop is None:
                sizes.append(-1)
            else:
                sizes.append(it.stop - starts[-1])
                if sizes[-1] == 0:
                    raise ValueError(
                        'The starts and ends of axis {} can not be equal, got {}:{}.'
                        .format(ix, starts[-1], it.stop))
            if it.step is not None:
                raise NotImplementedError
        elif isinstance(it, int):
            starts.append(it)
            sizes.append(0)
        else:
            raise TypeError('Unsupported type of indices: {}'.format(type(it).__name__))
    return starts, sizes


def reduce(data, axes=None, reduction='sum'):
    """Reduce data."""
    if reduction == 'sum':
        result = data.sum(axes)
    elif reduction == 'mean':
        result = data.mean(axes)
    else:
        result = data
    if not isinstance(result, np.ndarray):
        result = np.array(result, data.dtype)
    return result


def reduce_like(data, other, reduction='sum'):
    """Reduce data like the other."""
    while data.shape != other.shape:
        axis, keepdims = None, True
        for i in range(1, len(data.shape) + 1):
            if len(other.shape) - i < 0:
                axis, keepdims = -i, False
                break
            elif data.shape[-i] != other.shape[-i]:
                axis = -i
                break
        if axis is not None:
            if keepdims:
                assert other.shape[axis] == 1, 'Bad reduce case.'
            if reduction == 'sum':
                data = data.sum(axis=axis, keepdims=keepdims)
            elif reduction == 'mean':
                data = data.mean(axis=axis, keepdims=keepdims)
            else:
                raise ValueError('Unknown reduction:', reduction)
    return data


def uniform(shape, dtype='float32'):
    """Return the uniform data with given shape."""
    return np.random.uniform(size=shape).astype(dtype)


if __name__ == '__main__':
    run_tests()