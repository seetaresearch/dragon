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
"""Random API."""

from dragon.core.framework.config import set_random_seed as set_seed
from dragon.core.ops.random_ops import glorot_normal
from dragon.core.ops.random_ops import glorot_uniform
from dragon.core.ops.random_ops import multinomial
from dragon.core.ops.random_ops import permutation
from dragon.core.ops.random_ops import random_normal as normal
from dragon.core.ops.random_ops import random_normal_like as normal_like
from dragon.core.ops.random_ops import random_uniform as uniform
from dragon.core.ops.random_ops import random_uniform_like as uniform_like
from dragon.core.ops.random_ops import truncated_normal
