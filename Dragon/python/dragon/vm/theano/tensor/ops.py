# --------------------------------------------------------
# Theano for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.ops as ops

def dot(x1, x2, **kwargs):
    return ops.Dot([x1, x2], **kwargs)

def sigmoid(x, **kwargs):
    return ops.Sigmoid(x, **kwargs)

def tanh(x, **kwargs):
    return ops.Tanh(x, **kwargs)