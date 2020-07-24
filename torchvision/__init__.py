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
"""Datasets, Transforms and Models specific to Computer Vision."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

# Modules
from dragon.vm.torchvision._api import ops

# Attributes
_API_MODULE = ops
_current_module = _sys.modules[__name__]
_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
if not hasattr(_current_module, '__path__'):
    __path__ = [_api_dir]
elif _api_dir not in __path__:
    __path__.append(_api_dir)
__all__ = [_s for _s in dir() if not _s.startswith('_')]
