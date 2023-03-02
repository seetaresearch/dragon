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

# Functions
from dragon.core.device.mlu import current_device
from dragon.core.device.mlu import get_device_capability
from dragon.core.device.mlu import get_device_count
from dragon.core.device.mlu import get_device_name
from dragon.core.device.mlu import is_available
from dragon.core.device.mlu import memory_allocated
from dragon.core.device.mlu import set_cnnl_flags
from dragon.core.device.mlu import set_default_device
from dragon.core.device.mlu import set_device
from dragon.core.device.mlu import set_random_seed
from dragon.core.device.mlu import synchronize

__all__ = [_s for _s in dir() if not _s.startswith('_')]