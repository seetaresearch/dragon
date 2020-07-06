# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

# Modules
from dragon.vm.torch import jit
from dragon.vm.torch import nn
from dragon.vm.torch import onnx
from dragon.vm.torch import optim
from dragon.vm.torch import vision

# Classes
from dragon.vm.torch.autograd import Variable
from dragon.vm.torch.cpp import device
from dragon.vm.torch.cpp import Size
from dragon.vm.torch.tensor import ByteTensor
from dragon.vm.torch.tensor import CharTensor
from dragon.vm.torch.tensor import DoubleTensor
from dragon.vm.torch.tensor import FloatTensor
from dragon.vm.torch.tensor import HalfTensor
from dragon.vm.torch.tensor import IntTensor
from dragon.vm.torch.tensor import LongTensor
from dragon.vm.torch.tensor import Tensor

# Functions
from dragon.vm.torch.autograd import enable_grad
from dragon.vm.torch.autograd import no_grad
from dragon.vm.torch.autograd import set_grad_enabled
from dragon.vm.torch.cpp import from_numpy
from dragon.vm.torch.ops import tensorbind as _
from dragon.vm.torch.ops.array.functional import argmax
from dragon.vm.torch.ops.array.functional import argmin
from dragon.vm.torch.ops.array.functional import assign
from dragon.vm.torch.ops.array.functional import cast
from dragon.vm.torch.ops.array.functional import cat
from dragon.vm.torch.ops.array.functional import channel_normalize
from dragon.vm.torch.ops.array.functional import channel_shuffle
from dragon.vm.torch.ops.array.functional import chunk
from dragon.vm.torch.ops.array.functional import cumsum
from dragon.vm.torch.ops.array.functional import expand
from dragon.vm.torch.ops.array.functional import index_select
from dragon.vm.torch.ops.array.functional import masked_select
from dragon.vm.torch.ops.array.functional import masked_fill
from dragon.vm.torch.ops.array.functional import multinomial
from dragon.vm.torch.ops.array.functional import narrow
from dragon.vm.torch.ops.array.functional import max
from dragon.vm.torch.ops.array.functional import mean
from dragon.vm.torch.ops.array.functional import min
from dragon.vm.torch.ops.array.functional import nonzero
from dragon.vm.torch.ops.array.functional import one_hot
from dragon.vm.torch.ops.array.functional import permute
from dragon.vm.torch.ops.array.functional import repeat
from dragon.vm.torch.ops.array.functional import reshape
from dragon.vm.torch.ops.array.functional import split
from dragon.vm.torch.ops.array.functional import squeeze
from dragon.vm.torch.ops.array.functional import stack
from dragon.vm.torch.ops.array.functional import sum
from dragon.vm.torch.ops.array.functional import topk
from dragon.vm.torch.ops.array.functional import unsqueeze
from dragon.vm.torch.ops.array.functional import where
from dragon.vm.torch.ops.init.functional import arange
from dragon.vm.torch.ops.init.functional import eye
from dragon.vm.torch.ops.init.functional import normal
from dragon.vm.torch.ops.init.functional import ones
from dragon.vm.torch.ops.init.functional import ones_like
from dragon.vm.torch.ops.init.functional import rand
from dragon.vm.torch.ops.init.functional import randn
from dragon.vm.torch.ops.init.functional import uniform
from dragon.vm.torch.ops.init.functional import zeros
from dragon.vm.torch.ops.init.functional import zeros_like
from dragon.vm.torch.ops.math.functional import abs
from dragon.vm.torch.ops.math.functional import add
from dragon.vm.torch.ops.math.functional import axpby
from dragon.vm.torch.ops.math.functional import bitwise_not
from dragon.vm.torch.ops.math.functional import bitwise_xor
from dragon.vm.torch.ops.math.functional import ceil
from dragon.vm.torch.ops.math.functional import clamp
from dragon.vm.torch.ops.math.functional import cos
from dragon.vm.torch.ops.math.functional import div
from dragon.vm.torch.ops.math.functional import eq
from dragon.vm.torch.ops.math.functional import exp
from dragon.vm.torch.ops.math.functional import floor
from dragon.vm.torch.ops.math.functional import ge
from dragon.vm.torch.ops.math.functional import gt
from dragon.vm.torch.ops.math.functional import isinf
from dragon.vm.torch.ops.math.functional import isnan
from dragon.vm.torch.ops.math.functional import le
from dragon.vm.torch.ops.math.functional import log
from dragon.vm.torch.ops.math.functional import logsumexp
from dragon.vm.torch.ops.math.functional import lt
from dragon.vm.torch.ops.math.functional import maximum
from dragon.vm.torch.ops.math.functional import minimum
from dragon.vm.torch.ops.math.functional import mm
from dragon.vm.torch.ops.math.functional import mul
from dragon.vm.torch.ops.math.functional import ne
from dragon.vm.torch.ops.math.functional import neg
from dragon.vm.torch.ops.math.functional import pow
from dragon.vm.torch.ops.math.functional import reciprocal
from dragon.vm.torch.ops.math.functional import round
from dragon.vm.torch.ops.math.functional import rsqrt
from dragon.vm.torch.ops.math.functional import sign
from dragon.vm.torch.ops.math.functional import sin
from dragon.vm.torch.ops.math.functional import sqrt
from dragon.vm.torch.ops.math.functional import sub
from dragon.vm.torch.ops.metric.functional import topk_acc
from dragon.vm.torch.serialization import load
from dragon.vm.torch.serialization import save
from dragon.vm.torch.tensor import empty
from dragon.vm.torch.tensor import tensor
