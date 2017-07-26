# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = [
    'random_normal',
    'truncated_normal',
    'random_uniform'
]

import dragon.ops as ops
import dtypes

def random_normal(shape,
                  mean=0.0,
                  stddev=1.0,
                  dtype=dtypes.float32,
                  name=None):
    """
    Outputs random values from a normal distribution.

      Args:
        shape: A 1-D integer Python array. The shape of the output tensor.
        mean: A 0-D Python value of type `dtype`. The mean of the normal distribution.
        stddev: A 0-D Python value of type `dtype`. The standard deviation of the normal distribution.
        dtype: The type of the output.
        name: A name for the operation (optional).

      Returns:
        A tensor of the specified shape filled with random normal values.
    """

    return ops.RandomNormal(shape, mean, stddev, name=None)


def truncated_normal(shape,
                     mean=0.0,
                     stddev=1.0,
                     dtype=dtypes.float32,
                     name=None):
    """
    Outputs random values from a truncated normal distribution.

      The generated values follow a normal distribution with specified mean and
      standard deviation, except that values whose magnitude is more than 2 standard
      deviations from the mean are dropped and re-picked.

      Args:
        shape: A 1-D integer Python array. The shape of the output tensor.
        mean: A 0-D Python value of type `dtype`. The mean of the truncated normal distribution.
        stddev: A 0-D Python value of type `dtype`. The standard deviation f the truncated normal distribution.
        dtype: The type of the output.
        name: A name for the operation (optional).

      Returns:
        A tensor of the specified shape filled with random truncated normal values.
    """

    return ops.TruncatedNormal(shape, mean, stddev, name=name)


def random_uniform(shape,
                   minval=0,
                   maxval=None,
                   dtype=dtypes.float32,
                   name=None):
    """
    Outputs random values from a uniform distribution.

      The generated values follow a uniform distribution in the range
      `[minval, maxval)`. The lower bound `minval` is included in the range, while
      the upper bound `maxval` is excluded.

      For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
      be specified explicitly.

      In the integer case, the random integers are slightly biased unless
      `maxval - minval` is an exact power of two.  The bias is small for values of
      `maxval - minval` significantly smaller than the range of the output (either
      `2**32` or `2**64`).

      Args:
        shape: A 1-D integer Python array. The shape of the output tensor.
        minval: A 0-D Python value of type `dtype`. The lower bound on the
                range of random values to generate.  Defaults to 0.
        maxval: A 0-D Python value of type `dtype`. The upper bound on
                the range of random values to generate.  Defaults to 1 if `dtype` is floating point.
        dtype: The type of the output: `float32`, `float64`, `int32`, or `int64`.
        name: A name for the operation (optional).

      Returns:
        A tensor of the specified shape filled with random uniform values.

    """

    return ops.RandomUniform(shape, minval, maxval)