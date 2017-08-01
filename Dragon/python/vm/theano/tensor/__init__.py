# --------------------------------------------------------
# Theano for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor

from . import nnet
from . import ops

def matrix(name=None):
    if name is None: return Tensor().Variable()
    else: return Tensor(name).Variable()

def imatrix(name=None):
    return matrix(name)

def scalar(name=None):
    if name is None: return Tensor().Variable()
    else: return Tensor(name).Variable()

def iscalar(name=None):
    return scalar(name)

def grad(cost, wrt):
    """ append a grad tuple """
    grads = []
    if not isinstance(wrt, list): wrt = [wrt]
    for w in wrt:
        cost.grad_wrts.append(w.name)
        w.grad_objs.append(cost.name)
        grads.append(Tensor(w.name + '_grad'))
    if len(grads) == 1: return grads[0]
    return grads

dot = ops.dot
tanh = ops.tanh

