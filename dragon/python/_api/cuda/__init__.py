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
"""CUDA API"""

from dragon.core.device.cuda import Stream
from dragon.core.device.cuda import current_device
from dragon.core.device.cuda import get_device_capability
from dragon.core.device.cuda import get_device_count
from dragon.core.device.cuda import get_device_name
from dragon.core.device.cuda import is_available
from dragon.core.device.cuda import memory_allocated
from dragon.core.device.cuda import set_cublas_flags
from dragon.core.device.cuda import set_cudnn_flags
from dragon.core.device.cuda import set_default_device
from dragon.core.device.cuda import set_device
from dragon.core.device.cuda import set_random_seed
from dragon.core.device.cuda import synchronize
