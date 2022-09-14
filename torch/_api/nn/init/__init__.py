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
"""NN init module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

from dragon.vm.torch.core.nn.init import calculate_gain
from dragon.vm.torch.core.nn.init import constant_
from dragon.vm.torch.core.nn.init import dirac_
from dragon.vm.torch.core.nn.init import eye_
from dragon.vm.torch.core.nn.init import kaiming_normal_
from dragon.vm.torch.core.nn.init import kaiming_uniform_
from dragon.vm.torch.core.nn.init import normal_
from dragon.vm.torch.core.nn.init import ones_
from dragon.vm.torch.core.nn.init import trunc_normal_
from dragon.vm.torch.core.nn.init import uniform_
from dragon.vm.torch.core.nn.init import xavier_normal_
from dragon.vm.torch.core.nn.init import xavier_uniform_
from dragon.vm.torch.core.nn.init import zeros_

__all__ = [_s for _s in dir() if not _s.startswith('_')]
