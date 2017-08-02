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
from ..core import dtypes


def random_normal(shape,
                  mean=0.0,
                  stddev=1.0,
                  dtype=dtypes.float32,
                  name=None):

    return ops.RandomNormal(shape, mean, stddev, name=None)


def truncated_normal(shape,
                     mean=0.0,
                     stddev=1.0,
                     dtype=dtypes.float32,
                     name=None):

    return ops.TruncatedNormal(shape, mean, stddev, name=name)


def random_uniform(shape,
                   minval=0,
                   maxval=None,
                   dtype=dtypes.float32,
                   name=None):

    return ops.RandomUniform(shape, minval, maxval)