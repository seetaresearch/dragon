# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

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

import dragon.ops as ops
from ..core import dtypes


class Initializer(object):

    def __call__(self, shape, dtype=None):
        raise NotImplementedError


class Zeros(Initializer):

    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.Fill(shape, value=0.0)


class Ones(Initializer):

    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.Fill(shape, value=1.0)


class Constant(Initializer):

    def __init__(self, value=0, dtype=dtypes.float32):
        self.value = value
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.Fill(shape, value=self.value)


class RandomUniform(Initializer):

    def __init__(self, minval=0, maxval=1, dtype=dtypes.float32):
        self.minval = minval
        self.maxval = maxval
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.RandomUniform(shape, self.minval, self.maxval)


class RandomNormal(Initializer):

    def __init__(self, mean=0.0, stddev=1.0, dtype=dtypes.float32):
        self.mean = mean
        self.stddev = stddev
        assert dtype == dtypes.float32
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.RandomNormal(shape, self.mean, self.stddev)


class TruncatedNormal(Initializer):

    def __init__(self, mean=0.0, stddev=1.0, dtype=dtypes.float32):
        self.mean = mean
        self.stddev = stddev
        assert dtype == dtypes.float32
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.TruncatedNormal(shape, self.mean, self.stddev)


class VarianceScaling(Initializer):

    def __init__(self, scale=1.0,
                 mode="fan_in",
                 distribution="normal",
                 dtype=dtypes.float32):
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
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        if self.distribution == "normal":
            return ops.GlorotNormal(shape=shape, scale=self.scale, mode=self.mode)
        else:
            return ops.GlorotUniform(shape=shape, scale=self.scale, mode=self.mode)


zeros_initializer = Zeros
ones_initializer = Ones
constant_initializer = Constant
random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
variance_scaling_initializer = VarianceScaling


def glorot_uniform_initializer(dtype=dtypes.float32):

    return variance_scaling_initializer(scale=6.0,
                                        mode='fan_avg',
                                        distribution='uniform',
                                        dtype=dtype)


def glorot_normal_initializer(dtype=dtypes.float32):

    return variance_scaling_initializer(scale=2.0,
                                        mode='fan_avg',
                                        distribution='normal',
                                        dtype=dtype)