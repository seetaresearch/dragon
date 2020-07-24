# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
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

from dragon.vm.tensorlayer.core.initializers import Constant
from dragon.vm.tensorlayer.core.initializers import constant
from dragon.vm.tensorlayer.core.initializers import GlorotNormal
from dragon.vm.tensorlayer.core.initializers import glorot_normal
from dragon.vm.tensorlayer.core.initializers import GlorotUniform
from dragon.vm.tensorlayer.core.initializers import glorot_uniform
from dragon.vm.tensorlayer.core.initializers import Initializer
from dragon.vm.tensorlayer.core.initializers import Ones
from dragon.vm.tensorlayer.core.initializers import ones
from dragon.vm.tensorlayer.core.initializers import RandomNormal
from dragon.vm.tensorlayer.core.initializers import random_normal
from dragon.vm.tensorlayer.core.initializers import RandomUniform
from dragon.vm.tensorlayer.core.initializers import random_uniform
from dragon.vm.tensorlayer.core.initializers import TruncatedNormal
from dragon.vm.tensorlayer.core.initializers import truncated_normal
from dragon.vm.tensorlayer.core.initializers import Zeros
from dragon.vm.tensorlayer.core.initializers import zeros

__all__ = [_s for _s in dir() if not _s.startswith('_')]
