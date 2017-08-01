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
    """Initializer base class: all initializers inherit from this class."""

    def __call__(self, shape, dtype=None):
        raise NotImplementedError


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0."""

    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.Fill(shape, value=0.0)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.Fill(shape, value=1.0)


class Constant(Initializer):
    """Initializer that generates tensors with constant values."""

    def __init__(self, value=0, dtype=dtypes.float32):
        self.value = value
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.Fill(shape, value=self.value)


class RandomUniform(Initializer):
    """
    Initializer that generates tensors with a uniform distribution.

      Args:
        minval: A python scalar. Lower bound of the range
          of random values to generate.
        maxval: A python scalar. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        dtype: The data type.
    """

    def __init__(self, minval=0, maxval=1, dtype=dtypes.float32):
        self.minval = minval
        self.maxval = maxval
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.RandomUniform(shape, self.minval, self.maxval)


class RandomNormal(Initializer):
    """
    Initializer that generates tensors with a normal distribution.

      Args:
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        dtype: The data type. Only floating point types are supported.
    """

    def __init__(self, mean=0.0, stddev=1.0, dtype=dtypes.float32):
        self.mean = mean
        self.stddev = stddev
        assert dtype == dtypes.float32
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.RandomNormal(shape, self.mean, self.stddev)


class TruncatedNormal(Initializer):
    """
    Initializer that generates a truncated normal distribution.

      These values are similar to values from a `random_normal_initializer`
      except that values more than two standard deviations from the mean
      are discarded and re-drawn. This is the recommended initializer for
      neural network weights and filters.

      Args:
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        dtype: The data type. Only floating point types are supported.
    """

    def __init__(self, mean=0.0, stddev=1.0, dtype=dtypes.float32):
        self.mean = mean
        self.stddev = stddev
        assert dtype == dtypes.float32
        self.dtype = dtype

    def __call__(self, shape, dtype=None):
        if dtype is None: dtype = self.dtype
        return ops.TruncatedNormal(shape, self.mean, self.stddev)


class VarianceScaling(Initializer):
    """
    Initializer capable of adapting its scale to the shape of weights tensors.

      With `distribution="normal"`, samples are drawn from a truncated normal
      distribution centered on zero, with `stddev = sqrt(scale / n)`
      where n is:
        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

      With `distribution="uniform"`, samples are drawn from a uniform distribution
      within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

      Arguments:
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        dtype: The data type. Only floating point types are supported.
    """

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
    """
    The Glorot uniform initializer, also called Xavier uniform initializer.

      It draws samples from a uniform distribution within [-limit, limit]
      where `limit` is `sqrt(6 / (fan_in + fan_out))`
      where `fan_in` is the number of input units in the weight tensor
      and `fan_out` is the number of output units in the weight tensor.

      Reference: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

      Arguments:
        dtype: The data type. Only floating point types are supported.

      Returns:
        An initializer.
    """

    return variance_scaling_initializer(scale=6.0,
                                        mode='fan_avg',
                                        distribution='uniform',
                                        dtype=dtype)


def glorot_normal_initializer(dtype=dtypes.float32):
    """
    The Glorot normal initializer, also called Xavier normal initializer.

      It draws samples from a truncated normal distribution centered on 0
      with `stddev = sqrt(2 / (fan_in + fan_out))`
      where `fan_in` is the number of input units in the weight tensor
      and `fan_out` is the number of output units in the weight tensor.

      Reference: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

      Arguments:
        dtype: The data type. Only floating point types are supported.

      Returns:
        An initializer.
    """

    return variance_scaling_initializer(scale=2.0,
                                        mode='fan_avg',
                                        distribution='normal',
                                        dtype=dtype)