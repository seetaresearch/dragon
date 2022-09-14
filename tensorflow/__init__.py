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
"""An Open Source Machine Learning Framework for Everyone."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

# Modules
from dragon.vm.tensorflow._api import bitwise
from dragon.vm.tensorflow._api import dtypes
from dragon.vm.tensorflow._api import keras
from dragon.vm.tensorflow._api import linalg
from dragon.vm.tensorflow._api import math
from dragon.vm.tensorflow._api import nn
from dragon.vm.tensorflow._api import random
from dragon.vm.keras._api import initializers
from dragon.vm.keras._api import losses
from dragon.vm.keras._api import optimizers

# Classes
from dragon.vm.tensorflow.core.eager.backprop import GradientTape
from dragon.vm.tensorflow.core.framework.tensor_shape import TensorShape
from dragon.vm.tensorflow.core.framework.tensor_spec import TensorSpec
from dragon.vm.tensorflow.core.module.module import Module

# Functions
from dragon.vm.tensorflow.core.eager.def_function import function
from dragon.vm.tensorflow.core.framework.constant_op import constant
from dragon.vm.tensorflow.core.framework.dtypes import as_dtype
from dragon.vm.tensorflow.core.framework.dtypes import bfloat16
from dragon.vm.tensorflow.core.framework.dtypes import bool
from dragon.vm.tensorflow.core.framework.dtypes import complex128
from dragon.vm.tensorflow.core.framework.dtypes import complex64
from dragon.vm.tensorflow.core.framework.dtypes import double
from dragon.vm.tensorflow.core.framework.dtypes import DType
from dragon.vm.tensorflow.core.framework.dtypes import float16
from dragon.vm.tensorflow.core.framework.dtypes import float32
from dragon.vm.tensorflow.core.framework.dtypes import float64
from dragon.vm.tensorflow.core.framework.dtypes import half
from dragon.vm.tensorflow.core.framework.dtypes import int16
from dragon.vm.tensorflow.core.framework.dtypes import int32
from dragon.vm.tensorflow.core.framework.dtypes import int64
from dragon.vm.tensorflow.core.framework.dtypes import int8
from dragon.vm.tensorflow.core.framework.dtypes import qint16
from dragon.vm.tensorflow.core.framework.dtypes import qint32
from dragon.vm.tensorflow.core.framework.dtypes import qint8
from dragon.vm.tensorflow.core.framework.dtypes import quint16
from dragon.vm.tensorflow.core.framework.dtypes import quint8
from dragon.vm.tensorflow.core.framework.dtypes import string
from dragon.vm.tensorflow.core.framework.dtypes import uint16
from dragon.vm.tensorflow.core.framework.dtypes import uint32
from dragon.vm.tensorflow.core.framework.dtypes import uint64
from dragon.vm.tensorflow.core.framework.dtypes import uint8
from dragon.vm.tensorflow.core.framework.dtypes import variant
from dragon.vm.tensorflow.core.framework.ops import convert_to_tensor
from dragon.vm.tensorflow.core.framework.ops import device
from dragon.vm.tensorflow.core.framework.ops import name_scope
from dragon.vm.tensorflow.core.module.module import Module
from dragon.vm.tensorflow.core.ops.array_ops import broadcast_to
from dragon.vm.tensorflow.core.ops.array_ops import concat
from dragon.vm.tensorflow.core.ops.array_ops import expand_dims
from dragon.vm.tensorflow.core.ops.array_ops import fill
from dragon.vm.tensorflow.core.ops.array_ops import gather
from dragon.vm.tensorflow.core.ops.array_ops import identity
from dragon.vm.tensorflow.core.ops.array_ops import ones
from dragon.vm.tensorflow.core.ops.array_ops import ones_like
from dragon.vm.tensorflow.core.ops.array_ops import one_hot
from dragon.vm.tensorflow.core.ops.array_ops import pad
from dragon.vm.tensorflow.core.ops.array_ops import placeholder
from dragon.vm.tensorflow.core.ops.array_ops import reshape
from dragon.vm.tensorflow.core.ops.array_ops import reverse
from dragon.vm.tensorflow.core.ops.array_ops import roll
from dragon.vm.tensorflow.core.ops.array_ops import shape
from dragon.vm.tensorflow.core.ops.array_ops import slice
from dragon.vm.tensorflow.core.ops.array_ops import split
from dragon.vm.tensorflow.core.ops.array_ops import squeeze
from dragon.vm.tensorflow.core.ops.array_ops import tile
from dragon.vm.tensorflow.core.ops.array_ops import transpose
from dragon.vm.tensorflow.core.ops.array_ops import unique
from dragon.vm.tensorflow.core.ops.array_ops import unique_with_counts
from dragon.vm.tensorflow.core.ops.array_ops import unstack
from dragon.vm.tensorflow.core.ops.array_ops import zeros
from dragon.vm.tensorflow.core.ops.array_ops import zeros_like
from dragon.vm.tensorflow.core.ops.clip_ops import clip_by_value
from dragon.vm.tensorflow.core.ops.linalg_ops import eye
from dragon.vm.tensorflow.core.ops.math_ops import add
from dragon.vm.tensorflow.core.ops.math_ops import add_n
from dragon.vm.tensorflow.core.ops.math_ops import argmax
from dragon.vm.tensorflow.core.ops.math_ops import argmin
from dragon.vm.tensorflow.core.ops.math_ops import cast
from dragon.vm.tensorflow.core.ops.math_ops import divide
from dragon.vm.tensorflow.core.ops.math_ops import equal
from dragon.vm.tensorflow.core.ops.math_ops import exp
from dragon.vm.tensorflow.core.ops.math_ops import less
from dragon.vm.tensorflow.core.ops.math_ops import linspace
from dragon.vm.tensorflow.core.ops.math_ops import matmul
from dragon.vm.tensorflow.core.ops.math_ops import multiply
from dragon.vm.tensorflow.core.ops.math_ops import pow
from dragon.vm.tensorflow.core.ops.math_ops import range
from dragon.vm.tensorflow.core.ops.math_ops import reduce_mean
from dragon.vm.tensorflow.core.ops.math_ops import reduce_sum
from dragon.vm.tensorflow.core.ops.math_ops import sigmoid
from dragon.vm.tensorflow.core.ops.math_ops import sqrt
from dragon.vm.tensorflow.core.ops.math_ops import square
from dragon.vm.tensorflow.core.ops.math_ops import subtract
from dragon.vm.tensorflow.core.ops.math_ops import tanh
from dragon.vm.tensorflow.core.ops.sort_ops import argsort
from dragon.vm.tensorflow.core.ops.sort_ops import sort
from dragon.vm.tensorflow.core.ops.variables import Variable

# Attributes
_API_MODULES = [dtypes, keras]
_current_module = _sys.modules[__name__]
for _API_MODULE in _API_MODULES:
    _api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
    if not hasattr(_current_module, '__path__'):
        __path__ = [_api_dir]
    elif _api_dir not in __path__:
        __path__.append(_api_dir)
__all__ = [_s for _s in dir() if not _s.startswith('_')]
