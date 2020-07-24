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
"""Open standard for machine learning interoperability."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

# Classes
from dragon.vm.onnx.core.shell import Shell

# Attributes
from dragon.vm.onnx.core import nodes as _nodes
__all__ = [_s for _s in dir() if not _s.startswith('_')]
