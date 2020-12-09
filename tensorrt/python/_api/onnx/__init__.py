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

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

# Classes
from dragon.vm.onnx.core.backend.tensorrt import BackendRep

# Functions
from dragon.vm.onnx.core.backend.tensorrt import prepare as prepare_backend
from dragon.vm.onnx.core.backend.tensorrt import run_model
from dragon.vm.onnx.core.backend.tensorrt import run_node
from dragon.vm.onnx.core.backend.tensorrt import supports_device

# Attributes
__all__ = [_s for _s in dir() if not _s.startswith('_')]
