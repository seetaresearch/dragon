# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.ops as ops

__all__ = ['flatten']

def flatten(inputs, name=None):

    return ops.Flatten(inputs, axis=1)

