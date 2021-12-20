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
"""Optimizer module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

from dragon.vm.torch.core.optim.adam import Adam
from dragon.vm.torch.core.optim.adam import AdamW
from dragon.vm.torch.core.optim.optimizer import Optimizer
from dragon.vm.torch.core.optim.rmsprop import RMSprop
from dragon.vm.torch.core.optim.sgd import LARS
from dragon.vm.torch.core.optim.sgd import SGD

__all__ = [_s for _s in dir() if not _s.startswith('_')]
