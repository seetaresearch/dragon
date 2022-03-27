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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.distributed.backend import is_initialized
from dragon.core.distributed.backend import finalize
from dragon.core.distributed.backend import get_backend
from dragon.core.distributed.backend import get_group
from dragon.core.distributed.backend import get_rank
from dragon.core.distributed.backend import get_world_size
from dragon.core.distributed.backend import new_group
