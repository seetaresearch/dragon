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

from dragon.core.device.cuda import current_device
from dragon.core.device.cuda import enable_cudnn
from dragon.core.device.cuda import enable_cudnn_benchmark
from dragon.core.device.cuda import get_device_capability
from dragon.core.device.cuda import is_available
from dragon.core.device.cuda import set_default_device
from dragon.core.device.cuda import set_device
from dragon.core.device.cuda import synchronize

__all__ = [_s for _s in dir() if not _s.startswith('_')]
