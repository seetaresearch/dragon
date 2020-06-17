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

from dragon.vm.tensorflow.core.ops.init_ops import Constant
from dragon.vm.tensorflow.core.ops.init_ops import Constant as constant
from dragon.vm.tensorflow.core.ops.init_ops import GlorotNormal
from dragon.vm.tensorflow.core.ops.init_ops import GlorotNormal as glorot_normal
from dragon.vm.tensorflow.core.ops.init_ops import GlorotUniform
from dragon.vm.tensorflow.core.ops.init_ops import GlorotUniform as glorot_uniform
from dragon.vm.tensorflow.core.ops.init_ops import Initializer
from dragon.vm.tensorflow.core.ops.init_ops import Ones
from dragon.vm.tensorflow.core.ops.init_ops import Ones as ones
from dragon.vm.tensorflow.core.ops.init_ops import RandomNormal
from dragon.vm.tensorflow.core.ops.init_ops import RandomUniform
from dragon.vm.tensorflow.core.ops.init_ops import TruncatedNormal
from dragon.vm.tensorflow.core.ops.init_ops import VarianceScaling
from dragon.vm.tensorflow.core.ops.init_ops import Zeros
from dragon.vm.tensorflow.core.ops.init_ops import Zeros as zeros

__all__ = [_s for _s in dir() if not _s.startswith('_')]
