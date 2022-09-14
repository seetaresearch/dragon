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
"""A Computation Graph Virtual Machine Based Deep Learning Framework."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

# Modules
from dragon._api import autograph
from dragon._api import bitwise
from dragon._api import cuda
from dragon._api import distributed
from dragon._api import dlpack
from dragon._api import logging
from dragon._api import losses
from dragon._api import math
from dragon._api import metrics
from dragon._api import mps
from dragon._api import nn
from dragon._api import onnx
from dragon._api import optimizers
from dragon._api import random
from dragon._api import sysconfig
from dragon._api import vision

# Classes
from dragon.core.autograph.backprop import GradientTape
from dragon.core.framework.device_spec import DeviceSpec
from dragon.core.framework.tensor import Tensor
from dragon.core.framework.workspace import Workspace

# Functions
from dragon.core.autograph.context import eager_mode
from dragon.core.autograph.context import graph_mode
from dragon.core.autograph.function_lib import function
from dragon.core.framework.backend import load_library
from dragon.core.framework.config import get_num_threads
from dragon.core.framework.config import set_num_threads
from dragon.core.framework.context import device
from dragon.core.framework.context import name_scope
from dragon.core.framework.context import variable_scope
from dragon.core.framework.workspace import get_workspace
from dragon.core.framework.workspace import reset_workspace
from dragon.core.ops import tensor_ops as _
from dragon.core.ops.array_ops import assign
from dragon.core.ops.array_ops import boolean_mask
from dragon.core.ops.array_ops import broadcast_to
from dragon.core.ops.array_ops import concat
from dragon.core.ops.array_ops import expand_dims
from dragon.core.ops.array_ops import flatten
from dragon.core.ops.array_ops import gather
from dragon.core.ops.array_ops import gather_elements
from dragon.core.ops.array_ops import identity
from dragon.core.ops.array_ops import nonzero
from dragon.core.ops.array_ops import one_hot
from dragon.core.ops.array_ops import pad
from dragon.core.ops.array_ops import repeat
from dragon.core.ops.array_ops import reshape
from dragon.core.ops.array_ops import reverse
from dragon.core.ops.array_ops import roll
from dragon.core.ops.array_ops import scatter_add
from dragon.core.ops.array_ops import scatter_elements
from dragon.core.ops.array_ops import shape
from dragon.core.ops.array_ops import slice
from dragon.core.ops.array_ops import split
from dragon.core.ops.array_ops import squeeze
from dragon.core.ops.array_ops import stack
from dragon.core.ops.array_ops import tile
from dragon.core.ops.array_ops import transpose
from dragon.core.ops.array_ops import tril
from dragon.core.ops.array_ops import triu
from dragon.core.ops.array_ops import unstack
from dragon.core.ops.array_ops import unique
from dragon.core.ops.array_ops import where
from dragon.core.ops.constant_ops import constant
from dragon.core.ops.constant_ops import eye
from dragon.core.ops.constant_ops import eye_like
from dragon.core.ops.constant_ops import fill
from dragon.core.ops.constant_ops import linspace
from dragon.core.ops.constant_ops import ones
from dragon.core.ops.constant_ops import ones_like
from dragon.core.ops.constant_ops import range
from dragon.core.ops.constant_ops import zeros
from dragon.core.ops.constant_ops import zeros_like
from dragon.core.ops.framework_ops import python_plugin
from dragon.core.ops.framework_ops import stop_gradient
from dragon.core.ops.math_ops import cast
from dragon.core.ops.sort_ops import argsort
from dragon.core.ops.sort_ops import sort

# Version
from dragon.version import version as __version__

# Attributes
_API_MODULE = autograph
_current_module = _sys.modules[__name__]
_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
if not hasattr(_current_module, '__path__'):
    __path__ = [_api_dir]
elif _api_dir not in __path__:
    __path__.append(_api_dir)
__all__ = [_s for _s in dir() if not _s.startswith('_')]
