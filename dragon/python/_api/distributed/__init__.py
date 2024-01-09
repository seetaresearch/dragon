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
"""Distributed API."""

from dragon.core.distributed.backend import is_cncl_available
from dragon.core.distributed.backend import is_initialized
from dragon.core.distributed.backend import is_mpi_available
from dragon.core.distributed.backend import is_nccl_available
from dragon.core.distributed.backend import finalize
from dragon.core.distributed.backend import get_backend
from dragon.core.distributed.backend import get_group
from dragon.core.distributed.backend import get_rank
from dragon.core.distributed.backend import get_world_size
from dragon.core.distributed.backend import new_group
from dragon.core.ops.distributed_ops import all_gather
from dragon.core.ops.distributed_ops import all_reduce
from dragon.core.ops.distributed_ops import broadcast
from dragon.core.ops.distributed_ops import reduce_scatter
