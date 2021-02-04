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
"""Test the autograph module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import unittest

import dragon
from dragon.core.framework import config
from dragon.core.testing.unittest.common_utils import run_tests


class TestConfig(unittest.TestCase):
    """Test the graph config."""

    def test_execution(self):
        for mode in ('EAGER_MODE', 'GRAPH_MODE', 'UNKNOWN'):
            try:
                dragon.autograph.set_execution(mode)
                self.assertEqual(config.config().graph_execution, mode)
            except ValueError:
                pass

    def test_optimization(self):
        dragon.autograph.set_optimization(1)
        self.assertEqual(config.config().graph_optimization, 1)

    def test_scheduler(self):
        for scheduler in ('SIMPLE', 'FUSION', 'KNOWN', 'SIMPLE'):
            try:
                dragon.autograph.set_scheduler(scheduler)
                if scheduler == 'FUSION':
                    self.assertEqual(config.config().graph_type, 'FusionGraph')
                else:
                    self.assertEqual(config.config().graph_type, '')
            except ValueError:

                pass

    def test_verbosity(self):
        dragon.autograph.set_verbosity(1)
        self.assertEqual(config.config().graph_verbosity, 1)
        dragon.autograph.set_verbosity(0)


class TestFunction(unittest.TestCase):
    """Test the graph function."""

    @dragon.function(input_signature=[
        dragon.Tensor((1,), dtype='int32'),
        dragon.Tensor((1,), dtype='int32'),
        dragon.Tensor((1,), dtype='int32'),
    ])
    def func1(self, a, b, c=0, **kwargs):
        _ = kwargs
        return a + b + c

    def test_create_function(self):
        a = dragon.Tensor((), dtype='int32').set_value(1)
        b = dragon.Tensor((), dtype='int32').set_value(2)
        y = a + 1
        try:
            dragon.create_function(outputs=y, optimizer=dragon.optimizers.SGD())
        except ValueError:
            pass
        try:
            dragon.create_function(outputs=dragon.EagerTensor(1))
        except ValueError:
            pass
        try:
            f = dragon.create_function(outputs=y, givens={a: 1})
        except ValueError:
            f = dragon.create_function(outputs=y, givens={a: b})
        self.assertEqual(int(f()), 3)

    def test_def_function(self):
        @dragon.function(input_signature=[dragon.Tensor(None)])
        def func2(a, b):
            return a + b
        self.assertEqual(self.func1([1, 2], [3, 4]).get_value().tolist(), [4, 6])
        self.assertEqual(self.func1([1, 2], b=[3, 4]).get_value().tolist(), [4, 6])
        self.assertEqual(self.func1([1, 2], b=[3, 4], c=1).get_value().tolist(), [5, 7])
        self.assertEqual(self.func1([1, 2], b=[3, 4], c=1).get_value().tolist(), [5, 7])
        self.assertEqual(self.func1([1, 2], [3, 4], executing_stage='forward').get_value().tolist(), [4, 6])
        dragon.function(func=lambda: dragon.optimizers.SGD())()
        try:
            self.func1(1, 2, 3, 4)
        except ValueError:
            pass
        try:
            func2(1, 2)
        except ValueError:
            pass

    def test_update_function(self):
        optimizer = dragon.optimizers.SGD()
        try:
            _ = optimizer.op_type
        except KeyError:
            pass
        value = dragon.Tensor((), dtype='float32').set_value(1.)
        grad = dragon.Tensor((), dtype='float32').set_value(1.)
        optimizer.apply_gradients([(value, grad)])
        dragon.create_function(optimizer=optimizer)()


class TestOpSpec(unittest.TestCase):
    """Test the op spec."""

    sym1 = dragon.Tensor(None, None)
    sym2 = dragon.Tensor((1,))
    sym3 = dragon.Tensor((1, None))
    sym4 = dragon.Tensor((1, None, None, None))
    sym5 = dragon.Tensor((1, None, None, None, None))

    def test_accuracy(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.metrics.accuracy(
                [self.sym1, self.sym1]).shape, ())

    def test_arg_reduce(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.math.argmax(
                self.sym1, axis=0, keepdims=True).shape, None)
            self.assertEqual(dragon.math.argmax(
                self.sym1, axis=0, keepdims=False).shape, None)
            self.assertEqual(dragon.math.argmax(
                self.sym1, axis=None, keepdims=True).shape, (1,))
            self.assertEqual(dragon.math.argmax(
                self.sym1, axis=None, keepdims=False).shape, ())
            self.assertEqual(dragon.math.argmax(
                self.sym2, axis=0, keepdims=True).shape, (1,))
            self.assertEqual(dragon.math.argmax(
                self.sym2, axis=0, keepdims=False).shape, ())

    def test_binary_ops(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.math.add(
                [self.sym1, self.sym1]).shape, None)
            self.assertEqual(dragon.math.add(
                [self.sym2, self.sym2]).shape, (1,))
            self.assertEqual(dragon.math.add(
                [self.sym2, self.sym3]).shape, (1, None))
            self.assertEqual(dragon.math.add(
                [self.sym3, self.sym2]).shape, (1, None))
            self.assertEqual(dragon.math.equal(
                [self.sym1, self.sym1]).shape, None)

    def test_broadcast(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.broadcast_to(
                self.sym1, shape=(1,)).shape, None)
            self.assertEqual(dragon.broadcast_to(
                self.sym2, shape=(1, 2)).shape, (1, 2))
            self.assertEqual(dragon.broadcast_to(
                self.sym3, shape=(2,)).shape, self.sym3.shape[:-1] + (2,))
            self.assertEqual(dragon.broadcast_to(
                self.sym3, shape=(-1, 2, 2)).shape, (1, 2, 2))

    def test_cast(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.cast(self.sym1, 'float32').shape, None)

    def test_concat(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.concat([self.sym1, self.sym1]).shape, None)
            self.assertEqual(dragon.concat([self.sym1, self.sym2]).shape, (None,))
            self.assertEqual(dragon.concat([self.sym2, self.sym3], axis=0).shape, (2,))
            self.assertEqual(dragon.concat([self.sym2, self.sym3], axis=1).shape, None)

    def test_conv(self):
        w = dragon.Tensor((3, 3, 3, 3))
        with dragon.graph_mode():
            self.assertEqual(dragon.nn.conv2d(
                [self.sym1, self.sym1]).shape, None)
            self.assertEqual(dragon.nn.conv2d(
                [self.sym4, w]).shape, (self.sym4.shape[0], w.shape[0], None, None))
            self.assertEqual(dragon.nn.conv2d(
                [w, w], kernel_shape=1, out_channels=w.shape[0]).shape, w.shape)
            self.assertEqual(dragon.nn.conv2d(
                [w, w], kernel_shape=1, padding='SAME').shape, w.shape)
            self.assertEqual(dragon.nn.conv2d_transpose(
                [self.sym4, w], out_channels=w.shape[1]).shape,
                (self.sym4.shape[0], w.shape[1], None, None))
            self.assertEqual(dragon.nn.conv2d_transpose(
                [w, w], output_padding=(2, 2), kernel_shape=1).shape,
                (w.shape[0], w.shape[1], w.shape[2] + 2, w.shape[3] + 2))
            self.assertEqual(dragon.nn.conv2d_transpose(
                [w, w], output_shape=(4, 4), output_padding=(2, 2), kernel_shape=1).shape,
                (w.shape[0], w.shape[1], 6, 6))

    def test_depth_to_space(self):
        func1 = functools.partial(dragon.nn.depth_to_space, block_size=1)
        func2 = functools.partial(dragon.nn.space_to_depth, block_size=1)
        with dragon.graph_mode():
            for func in (func1, func2):
                self.assertEqual(func(self.sym1).shape, None)
                self.assertEqual(func(self.sym2).shape, None)
                self.assertEqual(func(self.sym4, data_format='NCHW').shape,
                                 (self.sym4.shape[0],) + (None,) * (len(self.sym4.shape) - 1))
                self.assertEqual(func(self.sym4, data_format='NCHW').shape,
                                 (self.sym4.shape[0],) + (None,) * (len(self.sym4.shape) - 1))
                self.assertEqual(func(dragon.Tensor((1, 2, 3)), data_format='NCHW').shape,
                                 dragon.Tensor((1, 2, 3)).shape)
                self.assertEqual(func(dragon.Tensor((1, 2, 3)), data_format='NHWC').shape,
                                 dragon.Tensor((1, 2, 3)).shape)

    def test_dot(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.math.dot(
                [self.sym1, self.sym1]).shape, None)
            self.assertEqual(dragon.math.dot(
                [self.sym2, self.sym2]).shape, ())
            self.assertEqual(dragon.math.dot(
                [dragon.Tensor(()), dragon.Tensor(())]).shape, ())
            self.assertEqual(dragon.math.dot(
                [self.sym3, self.sym3]).shape, (self.sym3.shape[0], self.sym3.shape[1]))
            self.assertEqual(dragon.math.dot(
                [self.sym3, self.sym2]).shape, self.sym3.shape[:-1])

    def test_eltwise_loss(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.losses.l2_loss(
                [self.sym1, self.sym1]).shape, ())
            self.assertEqual(dragon.losses.l2_loss(
                [self.sym1, self.sym1], reduction='none').shape, None)

    def test_expand_dims(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.expand_dims(
                self.sym1, axis=1).shape, None)
            self.assertEqual(dragon.expand_dims(
                self.sym2, axis=1).shape, (1, 1))
            self.assertEqual(dragon.expand_dims(
                self.sym2, axis=-1).shape, (1, 1))
            self.assertEqual(dragon.expand_dims(
                self.sym3, axis=0).shape, (1, 1, None))
            self.assertEqual(dragon.expand_dims(
                self.sym3, axis=(0, 3)).shape, (1, 1, None, 1))
            self.assertEqual(dragon.expand_dims(
                self.sym3, axis=(0, 3, 5)).shape, (1, 1, None, 1))

    def test_init_ops(self):
        init_funcs_v1 = [dragon.fill,
                         dragon.ones,
                         dragon.random.glorot_normal,
                         dragon.random.glorot_uniform,
                         dragon.random.normal,
                         dragon.random.uniform,
                         dragon.random.truncated_normal,
                         dragon.zeros]
        for func in init_funcs_v1:
            with dragon.graph_mode():
                self.assertEqual(func(shape=self.sym1.shape).shape, None)
                self.assertEqual(func(shape=self.sym2.shape).shape, self.sym2.shape)

    def test_flatten(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.flatten(
                self.sym1, axis=1).shape, None)
            self.assertEqual(dragon.flatten(
                self.sym1, keep_axes=2).shape, (None, None))
            self.assertEqual(dragon.flatten(
                self.sym2, keep_axes=2).shape, (1, None))
            self.assertEqual(dragon.flatten(
                self.sym4, keep_axes=2).shape, (1, None))
            self.assertEqual(dragon.flatten(
                self.sym4, axis=1, num_axes=3).shape, (1, None))
            self.assertEqual(dragon.flatten(
                self.sym4, axis=1, num_axes=-1).shape, (1, None))

    def test_gemm(self):
        w = dragon.Tensor((3, 2))
        with dragon.graph_mode():
            self.assertEqual(dragon.math.gemm(
                [self.sym1, w]).shape, None)
            self.assertEqual(dragon.math.gemm(
                [self.sym1, w], axis=1).shape, (None, 2))
            self.assertEqual(dragon.math.gemm(
                [self.sym1, self.sym1]).shape, None)

    def test_index_select(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.index_select(
                self.sym1, self.sym1).shape, None)
            self.assertEqual(dragon.index_select(
                self.sym1, self.sym2, axis=-1).shape, None)
            self.assertEqual(dragon.index_select(
                self.sym3, self.sym2, axis=1).shape, (1, 1))

    def test_linspace(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.linspace(
                start=1, stop=5, num=3).shape, (3,))
            self.assertEqual(dragon.linspace(
                start=(1, 2), stop=(3, 4), num=3, axis=1).shape, (2, 3))
            self.assertEqual(dragon.linspace(
                start=(1, 2), stop=(3, 4), num=3, axis=0).shape, (3, 2))

    def test_mask_select(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.masked_select(
                [self.sym1, self.sym1]).shape, (None,))

    def test_matmul(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.math.matmul(
                [self.sym1, self.sym1]).shape, None)
            self.assertEqual(dragon.math.matmul(
                [self.sym1, self.sym2]).shape, None)
            self.assertEqual(dragon.math.matmul(
                [self.sym1, self.sym3]).shape, None)
            self.assertEqual(dragon.math.matmul(
                [self.sym2, self.sym3]).shape, (None,))
            self.assertEqual(dragon.math.matmul(
                [self.sym3, self.sym2]).shape, (1,))
            self.assertEqual(dragon.math.matmul(
                [self.sym3, self.sym3]).shape, (1, None))
            self.assertEqual(dragon.math.matmul(
                [self.sym4, self.sym3]).shape, (1, None, None, None))
            self.assertEqual(dragon.math.matmul(
                [self.sym4, self.sym4]).shape, (1, None, None, None))

    def test_moments(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.math.moments(self.sym1)[0].shape, ())
            self.assertEqual(dragon.math.moments(self.sym1, axis=0)[0].shape, None)
            self.assertEqual(dragon.math.moments(self.sym1, keepdims=True)[0].shape, (1,))
            self.assertEqual(dragon.math.moments(self.sym2)[0].shape, ())
            self.assertEqual(dragon.math.moments(self.sym2, axis=0)[0].shape, ())
            self.assertEqual(dragon.math.moments(self.sym2, axis=1)[0].shape, (1,))
            self.assertEqual(dragon.math.moments(self.sym2, axis=0, keepdims=True)[0].shape, (1,))
            self.assertEqual(dragon.math.moments(dragon.Tensor(None, 'float64'))[0].dtype, 'float64')
            self.assertEqual(dragon.math.moments(dragon.Tensor(None, 'int64'))[0].dtype, 'float64')

    def test_multinomial(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.random.multinomial(self.sym1).shape, None)
            self.assertEqual(dragon.random.multinomial(self.sym2, num_samples=2).shape, (2,))

    def test_non_zero(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.nonzero(self.sym1).shape, None)
            self.assertEqual(dragon.nonzero(self.sym2).shape, (None, 1))

    def test_one_hot(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.one_hot(self.sym1, depth=2).shape, None)
            self.assertEqual(dragon.one_hot(self.sym2, depth=2).shape, (1, 2))

    def test_pad(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.pad(self.sym1, pads=[(1, 1)]).shape, None)
            self.assertEqual(dragon.pad(self.sym3, pads=[(1, 1)]).shape, (3, None))
            self.assertEqual(dragon.pad(self.sym3, pads=[(1, 1), (1, 1)]).shape, (3, None))

    def test_permutation(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.random.permutation(5).shape, (5,))

    def test_pool(self):
        func = functools.partial(dragon.nn.pool2d, kernel_shape=3, strides=1, pads=1)
        with dragon.graph_mode():
            self.assertEqual(func(self.sym1).shape, None)
            self.assertEqual(func(self.sym3).shape, (1, None))
            self.assertEqual(func(self.sym4).shape, (1, None, None, None))
            self.assertEqual(func(self.sym4, global_pool=True).shape, (1, None, 1, 1))
            self.assertEqual(func(dragon.Tensor((1, 3, 4, 4))).shape, (1, 3, 4, 4))
            self.assertEqual(func(dragon.Tensor((1, 3, 4, 4)), padding='SAME').shape, (1, 3, 4, 4))

    def test_predicative(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.math.is_inf(self.sym1).shape, self.sym1.shape)
            self.assertEqual(dragon.math.is_inf(self.sym3).shape, self.sym3.shape)
            self.assertEqual(dragon.math.is_nan(self.sym1).shape, self.sym1.shape)
            self.assertEqual(dragon.math.is_nan(self.sym3).shape, self.sym3.shape)

    def test_range(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.range(3).shape, (3,))
            self.assertEqual(dragon.range(3, 4).shape, (1,))
            self.assertEqual(dragon.range(3, delta=0).shape, None)

    def test_reduce(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.math.sum(self.sym1).shape, ())
            self.assertEqual(dragon.math.sum(self.sym1, axis=0).shape, None)
            self.assertEqual(dragon.math.sum(self.sym1, keepdims=True).shape, ())
            self.assertEqual(dragon.math.sum(self.sym2, axis=0).shape, ())
            self.assertEqual(dragon.math.sum(self.sym2, axis=1).shape, (1,))
            self.assertEqual(dragon.math.sum(self.sym2, axis=0, keepdims=True).shape, (1,))

    def test_repeat(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.repeat(self.sym1, axis=None, repeats=2).shape, (None,))
            self.assertEqual(dragon.repeat(self.sym1, axis=0, repeats=2).shape, None)
            self.assertEqual(dragon.repeat(self.sym2, axis=None, repeats=2).shape, (2,))
            self.assertEqual(dragon.repeat(self.sym3, axis=0, repeats=2).shape, (2, None))
            self.assertEqual(dragon.repeat(self.sym3, axis=1, repeats=2).shape, (1, None))

    def test_reshape(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.reshape(self.sym2, shape=(0, 1)).shape, (1, 1))
            self.assertEqual(dragon.reshape(self.sym3, shape=(0, -1)).shape, (1, None))
            self.assertEqual(dragon.reshape(self.sym3, shape=(0, 1, 0)).shape, None)

    def test_resize(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.vision.resize(
                self.sym4, sizes=(1,)).shape, (1, None, 1, 1))
            self.assertEqual(dragon.vision.resize(
                self.sym4, sizes=(1, 1)).shape, (1, None, 1, 1))
            self.assertEqual(dragon.vision.resize(
                self.sym4, sizes=(1, 1, 1, 1)).shape, (1, None, 1, 1))
            self.assertEqual(dragon.vision.resize(
                self.sym4, scales=(1,)).shape, (1, None, None, None))
            self.assertEqual(dragon.vision.resize(
                self.sym4, scales=(1, 1)).shape, (1, None, None, None))
            self.assertEqual(dragon.vision.resize(
                self.sym4, scales=(1, 1, 1, 1)).shape, (1, None, None, None))
            self.assertEqual(dragon.vision.resize(
                self.sym5, sizes=(1, 1, 1, 1)).shape, None)

    def test_roi_pool(self):
        rois = dragon.Tensor((2, 5))
        func = functools.partial(dragon.vision.roi_pool, pooled_h=7, pooled_w=7)
        with dragon.graph_mode():
            self.assertEqual(func([self.sym1, rois]).shape, None)
            self.assertEqual(func([self.sym4, rois]).shape, (2, None, 7, 7))
            self.assertEqual(func([self.sym4, self.sym1]).shape, (None, None, 7, 7))

    def test_slice(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.slice(self.sym1, (1,), (1,)).shape, None)
            self.assertEqual(dragon.slice(self.sym3, (1,), (1,)).shape, (1, None))

    def test_softmax_loss(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.losses.sparse_softmax_cross_entropy(
                [self.sym1, self.sym1]).shape, ())
            self.assertEqual(dragon.losses.sparse_softmax_cross_entropy(
                [self.sym1, self.sym1], reduction='none').shape, None)
            self.assertEqual(dragon.losses.sparse_softmax_cross_entropy(
                [self.sym3, self.sym1], reduction='none').shape, (self.sym3.shape[0],))

    def test_sort(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.sort(self.sym1)[0].shape, None)
            self.assertEqual(dragon.sort(self.sym2)[0].shape, self.sym2.shape)

    def test_split(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.split(self.sym1, 2)[0].shape, None)
            self.assertEqual(dragon.split(self.sym2, 2)[0].shape, (1,))
            self.assertEqual(dragon.split(self.sym2, 2, axis=1)[0].shape, None)
            self.assertEqual(dragon.split(self.sym2, (1, 1))[0].shape, (1,))
            self.assertEqual(dragon.split(self.sym2, 2, slice_points=(1,))[0].shape, (1,))
            self.assertEqual(dragon.split(self.sym3, 2, axis=1)[0].shape, (1, None))
            self.assertEqual(dragon.split(self.sym3, 2, axis=1, slice_points=(1,))[1].shape, (1, None))

    def test_squeeze(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.squeeze(self.sym1).shape, None)
            self.assertEqual(dragon.squeeze(self.sym2).shape, ())
            self.assertEqual(dragon.squeeze(self.sym2, axis=-1).shape, ())
            self.assertEqual(dragon.squeeze(self.sym3).shape, (None,))

    def test_stack(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.stack([self.sym1, self.sym1]).shape, None)
            self.assertEqual(dragon.stack([self.sym3, self.sym2]).shape, (2, 1, None))
            self.assertEqual(dragon.stack([self.sym3, self.sym3]).shape, (2, 1, None))
            self.assertEqual(dragon.stack([self.sym3, self.sym3], axis=-1).shape, (1, None, 2))

    def test_tile(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.tile(
                self.sym1, repeats=(1, 2)).shape, None)
            self.assertEqual(dragon.tile(
                self.sym3, repeats=(1, 2)).shape, (1, None))

    def test_topk(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.math.top_k(self.sym1)[0].shape, None)
            self.assertEqual(dragon.math.top_k(self.sym2, k=2)[0].shape, (2,))
            self.assertEqual(dragon.math.top_k(self.sym2, axis=1)[0].shape, None)

    def test_unchanged(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.math.negative(self.sym1).shape, None)

    def test_unique(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.unique(self.sym1).shape, (None,))
            self.assertEqual(dragon.unique(self.sym1, return_counts=True)[1].shape, (None,))
            self.assertEqual(dragon.unique(self.sym1, return_inverse=True)[1].shape, None)
            self.assertEqual(dragon.unique(self.sym1,
                                           return_inverse=True,
                                           return_counts=True)[1].shape, None)


class TestOpSpecWithTensorDesc(unittest.TestCase):
    """Test the op spec with tensor descriptors."""

    sym1 = dragon.Tensor(None)
    sym2 = dragon.Tensor((1, None))
    sym3 = dragon.Tensor((1, None, None, None))
    shape1 = dragon.shape(sym1)
    shape2 = [1, shape1, 1]

    def test_broadcast_to(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.broadcast_to(
                self.sym1, shape=self.shape1).shape, None)
            self.assertEqual(dragon.broadcast_to(
                self.sym2, shape=self.shape1).shape, (None,) * len(self.sym2.shape))
            self.assertEqual(dragon.broadcast_to(
                self.sym2, shape=self.shape2).shape, (None,) * len(self.shape2))

    def test_channel_normalize(self):
        func = functools.partial(dragon.channel_normalize,
                                 mean=(1., 1., 1.), std=(1., 1., 1.))
        with dragon.graph_mode():
            self.assertEqual(func(self.sym1).shape, None)
            self.assertEqual(func(self.sym1, perm=self.shape1).shape, None)
            self.assertEqual(func(self.sym2).shape, self.sym2.shape)
            self.assertEqual(func(self.sym2, perm=self.shape1).shape,
                             (None,) * len(self.sym2.shape))
            self.assertEqual(func(self.sym2, perm=self.shape2).shape,
                             (None,) * len(self.sym2.shape))

    def test_conv_transpose(self):
        w = dragon.Tensor((3, 3, 3, 3))
        with dragon.graph_mode():
            self.assertEqual(dragon.nn.conv2d_transpose(
                [self.sym1, self.sym1]).shape, None)
            self.assertEqual(dragon.nn.conv2d_transpose(
                [self.sym3, self.sym1]).shape, None)
            self.assertEqual(dragon.nn.conv2d_transpose(
                [self.sym3, w]).shape, (self.sym3.shape[0], w.shape[0], None, None))
            self.assertEqual(dragon.nn.conv2d_transpose(
                [w, w], output_padding=self.shape1).shape,
                (w.shape[0], w.shape[0], None, None))
            self.assertEqual(dragon.nn.conv2d_transpose(
                [w, w], output_padding=self.shape2).shape,
                (w.shape[0], w.shape[0], None, None))
            self.assertEqual(dragon.nn.conv2d_transpose(
                [w, w], output_shape=self.shape1).shape,
                (w.shape[0], w.shape[0], None, None))
            self.assertEqual(dragon.nn.conv2d_transpose(
                [w, w], output_shape=self.shape2).shape,
                (w.shape[0], w.shape[0], None, None))

    def test_init_ops(self):
        init_funcs_v1 = [dragon.fill,
                         dragon.ones,
                         dragon.random.glorot_normal,
                         dragon.random.glorot_uniform,
                         dragon.random.normal,
                         dragon.random.uniform,
                         dragon.random.truncated_normal,
                         dragon.zeros]
        init_funcs_v2 = [dragon.ones_like,
                         dragon.random.normal_like,
                         dragon.random.uniform_like,
                         dragon.zeros_like]
        for func in init_funcs_v1:
            with dragon.graph_mode():
                self.assertEqual(func(shape=self.shape1).shape, None)
                self.assertEqual(func(shape=self.shape2).shape, (None,) * len(self.shape2))
        for func in init_funcs_v2:
            with dragon.graph_mode():
                self.assertEqual(func(self.sym1).shape, None)
                self.assertEqual(func(self.sym2).shape, self.sym2.shape)

    def test_permutation(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.random.permutation(self.sym1).shape, (None,))

    def test_repeat(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.repeat(
                self.sym1, repeats=self.shape1).shape, None)
            self.assertEqual(dragon.repeat(
                self.sym2, repeats=self.shape1).shape, None)

    def test_reshape(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.reshape(
                self.sym1, shape=self.shape1).shape, None)
            self.assertEqual(dragon.reshape(
                self.sym2, shape=self.shape1).shape, None)
            self.assertEqual(dragon.reshape(
                self.sym2, shape=self.shape2).shape, (None,) * len(self.shape2))

    def test_resize(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.vision.resize(
                self.sym1, sizes=self.shape1).shape, None)
            self.assertEqual(dragon.vision.resize(
                self.sym1, scales=self.shape1).shape, None)
            self.assertEqual(dragon.vision.resize(
                self.sym2, sizes=self.shape1).shape, (None,) * len(self.sym2.shape))
            self.assertEqual(dragon.vision.resize(
                self.sym2, scales=self.shape1).shape, (None,) * len(self.sym2.shape))
            self.assertEqual(dragon.vision.resize(
                self.sym2, sizes=self.shape2).shape, (None,) * len(self.sym2.shape))
            self.assertEqual(dragon.vision.resize(
                self.sym2, scales=self.shape2).shape, (None,) * len(self.sym2.shape))

    def test_slice(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.slice(
                self.sym1, starts=self.shape1, sizes=self.shape1).shape, None)
            self.assertEqual(dragon.slice(
                self.sym2, starts=self.shape1, sizes=self.shape1).shape, None)
            self.assertEqual(dragon.slice(
                self.sym2, starts=self.shape2, sizes=self.shape2).shape, None)

    def test_tile(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.tile(
                self.sym1, repeats=self.shape1).shape, None)
            self.assertEqual(dragon.tile(
                self.sym2, repeats=self.shape1).shape, (None,) * len(self.sym2.shape))
            self.assertEqual(dragon.tile(
                self.sym2, repeats=self.shape2).shape, (None,) * len(self.sym2.shape))

    def test_transpose(self):
        with dragon.graph_mode():
            self.assertEqual(dragon.transpose(self.sym1).shape, None)
            self.assertEqual(dragon.transpose(self.sym1, perm=self.shape1).shape, None)
            self.assertEqual(dragon.transpose(self.sym2).shape, self.sym2.shape[::-1])
            self.assertEqual(dragon.transpose(
                self.sym2, perm=self.shape1).shape, (None,) * len(self.sym2.shape))
            self.assertEqual(dragon.transpose(
                self.sym2, perm=self.shape2).shape, (None,) * len(self.sym2.shape))


if __name__ == '__main__':
    run_tests()
