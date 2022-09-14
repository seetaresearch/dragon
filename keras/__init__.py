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
"""Keras: Deep Learning for humans."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

# Modules
from dragon.vm.keras._api import activations
from dragon.vm.keras._api import initializers
from dragon.vm.keras._api import layers
from dragon.vm.keras._api import losses
from dragon.vm.keras._api import optimizers
from dragon.vm.keras._api import regularizers

# Classes
from dragon.vm.keras.core.engine.sequential import Sequential

# Functions
from dragon.vm.keras.core.engine.input_layer import Input

# Attributes
_API_MODULES = [activations]
_current_module = _sys.modules[__name__]
for _API_MODULE in _API_MODULES:
    _api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
    if not hasattr(_current_module, '__path__'):
        __path__ = [_api_dir]
    elif _api_dir not in __path__:
        __path__.append(_api_dir)
__all__ = [_s for _s in dir() if not _s.startswith('_')]
