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
"""Tensors and Dynamic neural networks in Python with strong GPU acceleration."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

# Modules
from dragon.vm.torch._api import autograd
from dragon.vm.torch._api import distributed
from dragon.vm.torch._api import jit
from dragon.vm.torch._api import nn
from dragon.vm.torch._api import onnx
from dragon.vm.torch._api import optim
from dragon.vm.torch._api import utils

# Classes
from dragon.vm.torch.core.autograd.grad_mode import enable_grad
from dragon.vm.torch.core.autograd.grad_mode import no_grad
from dragon.vm.torch.core.autograd.grad_mode import set_grad_enabled
from dragon.vm.torch.core.cpp import device
from dragon.vm.torch.core.cpp import dtype
from dragon.vm.torch.core.cpp import Size
from dragon.vm.torch.core.tensor import BoolTensor
from dragon.vm.torch.core.tensor import ByteTensor
from dragon.vm.torch.core.tensor import CharTensor
from dragon.vm.torch.core.tensor import DoubleTensor
from dragon.vm.torch.core.tensor import FloatTensor
from dragon.vm.torch.core.tensor import HalfTensor
from dragon.vm.torch.core.tensor import IntTensor
from dragon.vm.torch.core.tensor import LongTensor
from dragon.vm.torch.core.tensor import Tensor

# Functions
from dragon.vm.torch.core.cpp import from_numpy
from dragon.vm.torch.core.ops import tensorbind as _
from dragon.vm.torch.core.ops.array.functional import argmax
from dragon.vm.torch.core.ops.array.functional import argmin
from dragon.vm.torch.core.ops.array.functional import assign
from dragon.vm.torch.core.ops.array.functional import cat
from dragon.vm.torch.core.ops.array.functional import channel_affine
from dragon.vm.torch.core.ops.array.functional import channel_normalize
from dragon.vm.torch.core.ops.array.functional import channel_shuffle
from dragon.vm.torch.core.ops.array.functional import chunk
from dragon.vm.torch.core.ops.array.functional import cumsum
from dragon.vm.torch.core.ops.array.functional import flatten
from dragon.vm.torch.core.ops.array.functional import index_select
from dragon.vm.torch.core.ops.array.functional import masked_select
from dragon.vm.torch.core.ops.array.functional import masked_fill
from dragon.vm.torch.core.ops.array.functional import multinomial
from dragon.vm.torch.core.ops.array.functional import narrow
from dragon.vm.torch.core.ops.array.functional import max
from dragon.vm.torch.core.ops.array.functional import mean
from dragon.vm.torch.core.ops.array.functional import min
from dragon.vm.torch.core.ops.array.functional import nonzero
from dragon.vm.torch.core.ops.array.functional import one_hot
from dragon.vm.torch.core.ops.array.functional import permute
from dragon.vm.torch.core.ops.array.functional import repeat
from dragon.vm.torch.core.ops.array.functional import reshape
from dragon.vm.torch.core.ops.array.functional import split
from dragon.vm.torch.core.ops.array.functional import squeeze
from dragon.vm.torch.core.ops.array.functional import stack
from dragon.vm.torch.core.ops.array.functional import sum
from dragon.vm.torch.core.ops.array.functional import topk
from dragon.vm.torch.core.ops.array.functional import unique
from dragon.vm.torch.core.ops.array.functional import unsqueeze
from dragon.vm.torch.core.ops.array.functional import where
from dragon.vm.torch.core.ops.init.functional import arange
from dragon.vm.torch.core.ops.init.functional import eye
from dragon.vm.torch.core.ops.init.functional import ones
from dragon.vm.torch.core.ops.init.functional import ones_like
from dragon.vm.torch.core.ops.init.functional import rand
from dragon.vm.torch.core.ops.init.functional import randn
from dragon.vm.torch.core.ops.init.functional import randperm
from dragon.vm.torch.core.ops.init.functional import zeros
from dragon.vm.torch.core.ops.init.functional import zeros_like
from dragon.vm.torch.core.ops.math.functional import abs
from dragon.vm.torch.core.ops.math.functional import add
from dragon.vm.torch.core.ops.math.functional import axpby
from dragon.vm.torch.core.ops.math.functional import bitwise_not
from dragon.vm.torch.core.ops.math.functional import bitwise_xor
from dragon.vm.torch.core.ops.math.functional import ceil
from dragon.vm.torch.core.ops.math.functional import clamp
from dragon.vm.torch.core.ops.math.functional import cos
from dragon.vm.torch.core.ops.math.functional import div
from dragon.vm.torch.core.ops.math.functional import eq
from dragon.vm.torch.core.ops.math.functional import exp
from dragon.vm.torch.core.ops.math.functional import floor
from dragon.vm.torch.core.ops.math.functional import ge
from dragon.vm.torch.core.ops.math.functional import gt
from dragon.vm.torch.core.ops.math.functional import isinf
from dragon.vm.torch.core.ops.math.functional import isnan
from dragon.vm.torch.core.ops.math.functional import le
from dragon.vm.torch.core.ops.math.functional import log
from dragon.vm.torch.core.ops.math.functional import logsumexp
from dragon.vm.torch.core.ops.math.functional import lt
from dragon.vm.torch.core.ops.math.functional import maximum
from dragon.vm.torch.core.ops.math.functional import minimum
from dragon.vm.torch.core.ops.math.functional import mm
from dragon.vm.torch.core.ops.math.functional import mul
from dragon.vm.torch.core.ops.math.functional import ne
from dragon.vm.torch.core.ops.math.functional import neg
from dragon.vm.torch.core.ops.math.functional import pow
from dragon.vm.torch.core.ops.math.functional import reciprocal
from dragon.vm.torch.core.ops.math.functional import round
from dragon.vm.torch.core.ops.math.functional import rsqrt
from dragon.vm.torch.core.ops.math.functional import sign
from dragon.vm.torch.core.ops.math.functional import sin
from dragon.vm.torch.core.ops.math.functional import sqrt
from dragon.vm.torch.core.ops.math.functional import sub
from dragon.vm.torch.core.serialization import load
from dragon.vm.torch.core.serialization import save
from dragon.vm.torch.core.tensor import empty
from dragon.vm.torch.core.tensor import tensor

# Aliases
bool = dtype('bool')
int8 = dtype('int8')
uint8 = dtype('uint8')
int16 = short = dtype('int16')
int32 = int = dtype('int32')
int64 = long = dtype('int64')
qint8 = dtype('qint8')
quint8 = dtype('quint8')
qint32 = dtype('qint32')
bfloat16 = dtype('bfloat16')
float16 = half = dtype('float16')
float32 = float = dtype('float32')
float64 = double = dtype('float64')
complex32 = dtype('complex32')
complex64 = dtype('complex64')
complex128 = dtype('complex128')

# Attributes
_API_MODULE = autograd
_current_module = _sys.modules[__name__]
_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
if not hasattr(_current_module, '__path__'):
    __path__ = [_api_dir]
elif _api_dir not in __path__:
    __path__.append(_api_dir)
__all__ = [_s for _s in dir() if not _s.startswith('_')]
