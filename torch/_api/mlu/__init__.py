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
"""MLU API."""

from dragon.vm.torch.core.backends.mlu import current_device
from dragon.vm.torch.core.backends.mlu import device_count
from dragon.vm.torch.core.backends.mlu import get_device_capability
from dragon.vm.torch.core.backends.mlu import get_device_name
from dragon.vm.torch.core.backends.mlu import is_available
from dragon.vm.torch.core.backends.mlu import manual_seed
from dragon.vm.torch.core.backends.mlu import manual_seed_all
from dragon.vm.torch.core.backends.mlu import memory_allocated
from dragon.vm.torch.core.backends.mlu import set_device
from dragon.vm.torch.core.backends.mlu import synchronize
