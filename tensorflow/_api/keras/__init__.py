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
"""Keras API."""

import os as _os

from dragon.vm.keras._api import activations
from dragon.vm.keras._api import initializers
from dragon.vm.keras._api import layers
from dragon.vm.keras._api import losses
from dragon.vm.keras._api import optimizers
from dragon.vm.keras._api import regularizers

from dragon.vm.keras.core.engine.sequential import Sequential
from dragon.vm.keras.core.engine.input_layer import Input

_api_dir = _os.path.dirname(_os.path.dirname(activations.__file__))
__path__.append(_api_dir) if _api_dir not in __path__ else None
__all__ = [_s for _s in dir() if not _s.startswith("_")]
