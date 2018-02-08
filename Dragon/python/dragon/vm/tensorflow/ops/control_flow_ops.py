# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = ['equal']

import dragon.ops as ops


def equal(a, b, name=None):

    return ops.Equal([a, b])