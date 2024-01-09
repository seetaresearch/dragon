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
"""ONNX API."""

from dragon.vm.onnx.core.backend.native import BackendRep
from dragon.vm.onnx.core.backend.native import prepare as prepare_backend
from dragon.vm.onnx.core.backend.native import run_model
from dragon.vm.onnx.core.backend.native import supports_device
from dragon.vm.onnx.core.frontend.native import export
from dragon.vm.onnx.core.frontend.native import record
