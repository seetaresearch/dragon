# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Tensors and Dynamic neural networks in Python with strong GPU acceleration."""

import os as _os

from dragon.vm.torch._api import autograd
from dragon.vm.torch._api import backends
from dragon.vm.torch._api import cuda
from dragon.vm.torch._api import distributed
from dragon.vm.torch._api import jit
from dragon.vm.torch._api import mlu
from dragon.vm.torch._api import mps
from dragon.vm.torch._api import nn
from dragon.vm.torch._api import onnx
from dragon.vm.torch._api import optim
from dragon.vm.torch._api import utils

from dragon.vm.torch.core.autograd.grad_mode import enable_grad
from dragon.vm.torch.core.autograd.grad_mode import inference_mode
from dragon.vm.torch.core.autograd.grad_mode import no_grad
from dragon.vm.torch.core.autograd.grad_mode import set_grad_enabled
from dragon.vm.torch.core.cpp import device
from dragon.vm.torch.core.cpp import dtype
from dragon.vm.torch.core.cpp import Size
from dragon.vm.torch.core.tensor import BFloat16Tensor
from dragon.vm.torch.core.tensor import BoolTensor
from dragon.vm.torch.core.tensor import ByteTensor
from dragon.vm.torch.core.tensor import CharTensor
from dragon.vm.torch.core.tensor import DoubleTensor
from dragon.vm.torch.core.tensor import FloatTensor
from dragon.vm.torch.core.tensor import HalfTensor
from dragon.vm.torch.core.tensor import IntTensor
from dragon.vm.torch.core.tensor import LongTensor
from dragon.vm.torch.core.tensor import Tensor

from dragon.vm.torch.core.ops import tensor_ops as _
from dragon.vm.torch.core.ops.array_ops import broadcast_to
from dragon.vm.torch.core.ops.array_ops import cat
from dragon.vm.torch.core.ops.array_ops import chunk
from dragon.vm.torch.core.ops.array_ops import flatten
from dragon.vm.torch.core.ops.array_ops import flip
from dragon.vm.torch.core.ops.array_ops import fliplr
from dragon.vm.torch.core.ops.array_ops import flipud
from dragon.vm.torch.core.ops.array_ops import gather
from dragon.vm.torch.core.ops.array_ops import index_select
from dragon.vm.torch.core.ops.array_ops import masked_select
from dragon.vm.torch.core.ops.array_ops import masked_fill
from dragon.vm.torch.core.ops.array_ops import multinomial
from dragon.vm.torch.core.ops.array_ops import narrow
from dragon.vm.torch.core.ops.array_ops import nonzero
from dragon.vm.torch.core.ops.array_ops import permute
from dragon.vm.torch.core.ops.array_ops import reshape
from dragon.vm.torch.core.ops.array_ops import roll
from dragon.vm.torch.core.ops.array_ops import scatter
from dragon.vm.torch.core.ops.array_ops import scatter_add
from dragon.vm.torch.core.ops.array_ops import split
from dragon.vm.torch.core.ops.array_ops import squeeze
from dragon.vm.torch.core.ops.array_ops import stack
from dragon.vm.torch.core.ops.array_ops import tile
from dragon.vm.torch.core.ops.array_ops import transpose
from dragon.vm.torch.core.ops.array_ops import tril
from dragon.vm.torch.core.ops.array_ops import triu
from dragon.vm.torch.core.ops.array_ops import unbind
from dragon.vm.torch.core.ops.array_ops import unique
from dragon.vm.torch.core.ops.array_ops import unsqueeze
from dragon.vm.torch.core.ops.array_ops import where
from dragon.vm.torch.core.ops.constant_ops import arange
from dragon.vm.torch.core.ops.constant_ops import as_tensor
from dragon.vm.torch.core.ops.constant_ops import empty
from dragon.vm.torch.core.ops.constant_ops import eye
from dragon.vm.torch.core.ops.constant_ops import from_numpy
from dragon.vm.torch.core.ops.constant_ops import full
from dragon.vm.torch.core.ops.constant_ops import full_like
from dragon.vm.torch.core.ops.constant_ops import linspace
from dragon.vm.torch.core.ops.constant_ops import ones
from dragon.vm.torch.core.ops.constant_ops import ones_like
from dragon.vm.torch.core.ops.constant_ops import tensor
from dragon.vm.torch.core.ops.constant_ops import zeros
from dragon.vm.torch.core.ops.constant_ops import zeros_like
from dragon.vm.torch.core.ops.math_ops import abs
from dragon.vm.torch.core.ops.math_ops import add
from dragon.vm.torch.core.ops.math_ops import addmm
from dragon.vm.torch.core.ops.math_ops import argmax
from dragon.vm.torch.core.ops.math_ops import argmin
from dragon.vm.torch.core.ops.math_ops import atan2
from dragon.vm.torch.core.ops.math_ops import baddbmm
from dragon.vm.torch.core.ops.math_ops import bitwise_and
from dragon.vm.torch.core.ops.math_ops import bitwise_not
from dragon.vm.torch.core.ops.math_ops import bitwise_or
from dragon.vm.torch.core.ops.math_ops import bitwise_xor
from dragon.vm.torch.core.ops.math_ops import bmm
from dragon.vm.torch.core.ops.math_ops import ceil
from dragon.vm.torch.core.ops.math_ops import clamp
from dragon.vm.torch.core.ops.math_ops import cos
from dragon.vm.torch.core.ops.math_ops import cummax
from dragon.vm.torch.core.ops.math_ops import cummin
from dragon.vm.torch.core.ops.math_ops import cumsum
from dragon.vm.torch.core.ops.math_ops import div
from dragon.vm.torch.core.ops.math_ops import eq
from dragon.vm.torch.core.ops.math_ops import exp
from dragon.vm.torch.core.ops.math_ops import floor
from dragon.vm.torch.core.ops.math_ops import ge
from dragon.vm.torch.core.ops.math_ops import gt
from dragon.vm.torch.core.ops.math_ops import isfinite
from dragon.vm.torch.core.ops.math_ops import isinf
from dragon.vm.torch.core.ops.math_ops import isnan
from dragon.vm.torch.core.ops.math_ops import le
from dragon.vm.torch.core.ops.math_ops import log
from dragon.vm.torch.core.ops.math_ops import logical_and
from dragon.vm.torch.core.ops.math_ops import logical_not
from dragon.vm.torch.core.ops.math_ops import logical_or
from dragon.vm.torch.core.ops.math_ops import logical_xor
from dragon.vm.torch.core.ops.math_ops import logsumexp
from dragon.vm.torch.core.ops.math_ops import lt
from dragon.vm.torch.core.ops.math_ops import matmul
from dragon.vm.torch.core.ops.math_ops import max
from dragon.vm.torch.core.ops.math_ops import maximum
from dragon.vm.torch.core.ops.math_ops import mean
from dragon.vm.torch.core.ops.math_ops import min
from dragon.vm.torch.core.ops.math_ops import minimum
from dragon.vm.torch.core.ops.math_ops import mm
from dragon.vm.torch.core.ops.math_ops import mul
from dragon.vm.torch.core.ops.math_ops import nan_to_num
from dragon.vm.torch.core.ops.math_ops import ne
from dragon.vm.torch.core.ops.math_ops import neg
from dragon.vm.torch.core.ops.math_ops import norm
from dragon.vm.torch.core.ops.math_ops import pow
from dragon.vm.torch.core.ops.math_ops import reciprocal
from dragon.vm.torch.core.ops.math_ops import round
from dragon.vm.torch.core.ops.math_ops import rsqrt
from dragon.vm.torch.core.ops.math_ops import sigmoid
from dragon.vm.torch.core.ops.math_ops import sign
from dragon.vm.torch.core.ops.math_ops import sin
from dragon.vm.torch.core.ops.math_ops import sqrt
from dragon.vm.torch.core.ops.math_ops import square
from dragon.vm.torch.core.ops.math_ops import sub
from dragon.vm.torch.core.ops.math_ops import sum
from dragon.vm.torch.core.ops.math_ops import tanh
from dragon.vm.torch.core.ops.math_ops import var
from dragon.vm.torch.core.ops.math_ops import var_mean
from dragon.vm.torch.core.ops.random_ops import normal
from dragon.vm.torch.core.ops.random_ops import rand
from dragon.vm.torch.core.ops.random_ops import randn
from dragon.vm.torch.core.ops.random_ops import randperm
from dragon.vm.torch.core.ops.sort_ops import argsort
from dragon.vm.torch.core.ops.sort_ops import sort
from dragon.vm.torch.core.ops.sort_ops import topk
from dragon.vm.torch.core.random import manual_seed
from dragon.vm.torch.core.serialization import load
from dragon.vm.torch.core.serialization import save
from dragon.vm.torch.core.tensor import set_default_tensor_type

# Aliases
bool = dtype("bool")
int8 = dtype("int8")
uint8 = dtype("uint8")
int16 = short = dtype("int16")
int32 = int = dtype("int32")
int64 = long = dtype("int64")
qint8 = dtype("qint8")
quint8 = dtype("quint8")
qint32 = dtype("qint32")
bfloat16 = dtype("bfloat16")
float16 = half = dtype("float16")
float32 = float = dtype("float32")
float64 = double = dtype("float64")
complex32 = dtype("complex32")
complex64 = dtype("complex64")
complex128 = dtype("complex128")

# Attributes
_api_dir = _os.path.dirname(_os.path.dirname(autograd.__file__))
__path__.append(_api_dir) if _api_dir not in __path__ else None
__all__ = [_s for _s in dir() if not _s.startswith("_")]
