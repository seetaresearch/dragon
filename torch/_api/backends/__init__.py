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
"""Backends module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

# Modules
from dragon.vm.torch.core.backends import cnnl
from dragon.vm.torch.core.backends import cuda
from dragon.vm.torch.core.backends import cudnn
from dragon.vm.torch.core.backends import mlu
from dragon.vm.torch.core.backends import mps
from dragon.vm.torch.core.backends import openmp

__all__ = [_s for _s in dir() if not _s.startswith('_')]