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
"""Checkpoint utility module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

from dragon.vm.torch.core.utils.checkpoint import checkpoint
from dragon.vm.torch.core.utils.checkpoint import checkpoint_sequential
from dragon.vm.torch.core.utils.checkpoint import no_checkpoint

__all__ = [_s for _s in dir() if not _s.startswith('_')]
