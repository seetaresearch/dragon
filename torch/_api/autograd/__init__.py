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
"""Autograd module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

# Classes
from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.autograd.grad_mode import enable_grad
from dragon.vm.torch.core.autograd.grad_mode import no_grad
from dragon.vm.torch.core.autograd.grad_mode import set_grad_enabled
from dragon.vm.torch.core.autograd.variable import Variable

# Functions
from dragon.vm.torch.core.autograd.functional import backward

__all__ = [_s for _s in dir() if not _s.startswith('_')]
