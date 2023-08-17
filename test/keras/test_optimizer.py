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
"""Test optimizer module."""

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

    def __init__(self, method_name="runTest"):
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
            inputs[i] = x.numpy() if hasattr(x, "numpy") else x
        first = inputs[:num_first] if num_first > 1 else inputs[0]
        second = inputs[num_first : len(inputs)] if num_second > 1 else inputs[num_first]
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


class TestOptimizer(OpTestCase):
    """Test optimizer."""

    def test_adam_optimizer(self):
        optimizer = tf.keras.optimizers.Adam()
        lr, eps = optimizer.learning_rate, optimizer.epsilon
        beta1, beta2 = optimizer.beta_1, optimizer.beta_2
        data1 = uniform((2, 3))
        data2, data3 = np.zeros((2, 3), "float32"), np.zeros((2, 3), "float32")
        param = new_tensor(data1)
        tf.keras.regularizers.get("l2")(param)
        tf.keras.regularizers.get("l1_l2")(param)
        tf.keras.regularizers.L1(l1=0)(param)
        tf.keras.regularizers.L2(l2=0)(param)
        tf.keras.regularizers.L1L2(l1=0, l2=0)(param)
        tf.keras.regularizers.l1_l2(l1=0, l2=0)(param)
        self.assertEqual(getattr(param, "_weight_decay"), 0.0)
        for i in range(3):
            t = i + 1
            coef = math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
            data4 = uniform((2, 3))
            grad = new_tensor(data4)
            optimizer.apply_gradients([[grad, param]])
            data2 = beta1 * data2 + (1 - beta1) * data4
            data3 = beta2 * data3 + (1 - beta2) * np.square(data4)
            data1 -= lr * coef * data2 / (np.sqrt(data3) + eps)
            self.assertEqual(param, data1)
        self.assertEqual(optimizer.iterations, 3)
        optimizer.iterations = 233
        self.assertEqual(optimizer.iterations, 233)

    def test_nesterov_optimizer(self):
        optimizer = tf.keras.optimizers.SGD(nesterov=True)
        momentum, lr = optimizer.momentum, optimizer.learning_rate
        data1, data2 = uniform((2, 3)), np.zeros((2, 3), "float32")
        param = new_tensor(data1)
        for i in range(3):
            data3 = uniform((2, 3))
            grad = new_tensor(data3)
            optimizer.apply_gradients([[grad, param]])
            data2 = momentum * data2 + data3
            data1 -= lr * (momentum * data2 + data3)
            self.assertEqual(param, data1)

    def test_rmsprop_update(self):
        optimizer = tf.keras.optimizers.RMSprop()
        momentum, lr = optimizer.momentum, optimizer.learning_rate
        alpha, eps = optimizer.rho, optimizer.epsilon
        data1 = uniform((2, 3))
        data2, data3 = np.zeros((2, 3), "float32"), np.zeros((2, 3), "float32")
        param = new_tensor(data1)
        for i in range(3):
            data4 = uniform((2, 3))
            grad = new_tensor(data4)
            optimizer.apply_gradients([[grad, param]])
            data3 = alpha * data3 + (1 - alpha) * np.square(data4)
            data2 = momentum * data2 + (data4 / (np.sqrt(data3) + eps))
            data1 -= lr * data2
            self.assertEqual(param, data1)

    def test_sgd_optimizer(self):
        optimizer = tf.keras.optimizers.SGD()
        momentum, lr = optimizer.momentum, optimizer.learning_rate
        data1, data2 = uniform((2, 3)), np.zeros((2, 3), "float32")
        param = new_tensor(data1)
        for i in range(3):
            data3 = uniform((2, 3))
            grad = new_tensor(data3)
            optimizer.apply_gradients([[grad, param]])
            data2 = momentum * data2 + data3
            data1 -= lr * data2
            self.assertEqual(param, data1)


def new_tensor(data):
    """Create a new tensor for current execution."""
    return tf.constant(data)


def uniform(shape, dtype="float32"):
    """Return the uniform data with given shape."""
    return np.random.uniform(-1.0, 1.0, size=shape).astype(dtype)


if __name__ == "__main__":
    run_tests()
