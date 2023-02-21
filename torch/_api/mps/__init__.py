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
"""MPS module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

# Functions
from dragon.vm.torch.core.backends.mps import current_device
from dragon.vm.torch.core.backends.mps import device_count
from dragon.vm.torch.core.backends.mps import get_device_family
from dragon.vm.torch.core.backends.mps import get_device_name
from dragon.vm.torch.core.backends.mps import is_available
from dragon.vm.torch.core.backends.mps import is_built
from dragon.vm.torch.core.backends.mps import manual_seed
from dragon.vm.torch.core.backends.mps import manual_seed_all
from dragon.vm.torch.core.backends.mps import memory_allocated
from dragon.vm.torch.core.backends.mps import set_device
from dragon.vm.torch.core.backends.mps import synchronize

__all__ = [_s for _s in dir() if not _s.startswith('_')]
