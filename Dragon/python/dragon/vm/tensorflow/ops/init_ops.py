# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon

from dragon.vm.tensorflow.framework import dtypes


__all__ = [
    'zeros_initializer',
    'ones_initializer',
    'constant_initializer',
    'random_uniform_initializer',
    'random_normal_initializer',
    'truncated_normal_initializer',
    'variance_scaling_initializer',
    'glorot_uniform_initializer',
    'glorot_normal_initializer',
]


class Initializer(object):
    """The basic Initializer."""

    def __call__(self, shape, dtype=None, **kwargs):
        raise NotImplementedError


class Zeros(Initializer):
    """The initializer that sets tensors to 0.

    Parameters
    ----------
    shape : list, tuple or Tensor
        The shape of the initializer.
    dtype : DType
        The data type.

    Returns
    -------
    Tensor
        The initializer.

    """
    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None: dtype = self.dtype
        return dragon.ops.Fill(shape, value=0, dtype=dtype.name)


class Ones(Initializer):
    """The initializer that sets tensors to 1.

    Parameters
    ----------
    shape : list, tuple or Tensor
        The shape of the initializer.
    dtype : DType
        The data type.

    Returns
    -------
    Tensor
        The initializer.

    """
    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None: dtype = self.dtype
        return dragon.ops.Fill(shape, value=1, dtype=dtype.name)


class Constant(Initializer):
    def __init__(self, value=0, dtype=dtypes.float32):
        self.value = value
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None: dtype = self.dtype
        return dragon.ops.Fill(shape, value=self.value, dtype=dtype.name)


class RandomUniform(Initializer):
    def __init__(self, minval=0, maxval=1, dtype=dtypes.float32):
        self.minval = minval
        self.maxval = maxval
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None: dtype = self.dtype
        return dragon.ops.RandomUniform(
            shape, self.minval, self.maxval, dtype=dtype.name)


class RandomNormal(Initializer):
    def __init__(self, mean=0.0, stddev=1.0, dtype=dtypes.float32):
        self.mean = mean
        self.stddev = stddev
        assert dtype == dtypes.float32
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None: dtype = self.dtype
        return dragon.ops.RandomNormal(
            shape, self.mean, self.stddev, dtype=dtype.name)


class TruncatedNormal(Initializer):
    def __init__(self, mean=0.0, stddev=1.0, dtype=dtypes.float32):
        self.mean = mean
        self.stddev = stddev
        assert dtype == dtypes.float32
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None: dtype = self.dtype
        return dragon.ops.TruncatedNormal(
            shape, self.mean, self.stddev, dtype=dtype.name)


class VarianceScaling(Initializer):
    def __init__(self,
        scale=1.0, mode="fan_in",
            distribution="normal",
                 dtype=dtypes.float32
    ):
        if scale <= 0.:
            raise ValueError("`scale` must be positive float.")

        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError("Invalid `mode` argument:", mode)

        distribution = distribution.lower()
        if distribution not in {"normal", "uniform"}:
            raise ValueError("Invalid `distribution` argument:", distribution)

        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        assert dtype == dtypes.float32
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None: dtype = self.dtype
        if self.distribution == "normal":
            return dragon.ops.GlorotNormal(shape=shape, scale=self.scale * 2.,
                mode=self.mode, dtype=dtype.name)
        else:
            return dragon.ops.GlorotUniform(shape=shape, scale=self.scale * 3.,
                mode=self.mode, dtype=dtype.name)


zeros_initializer = Zeros
ones_initializer = Ones
constant_initializer = Constant
random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
variance_scaling_initializer = VarianceScaling


def glorot_uniform_initializer(dtype=dtypes.float32):
    return variance_scaling_initializer(scale=1.0,
        mode='fan_avg', distribution='uniform', dtype=dtype)


def glorot_normal_initializer(dtype=dtypes.float32):
    return variance_scaling_initializer(scale=1.0,
        mode='fan_avg', distribution='normal', dtype=dtype)