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
"""DLPack utilities."""

from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor


def from_dlpack(dlpack):
    """Create a tensor sharing the dlpack data.

    Parameters
    ----------
    dlpack : PyCapsule
        The capsule object of a dlpack tensor.

    Returns
    -------
    dragon.Tensor
        The tensor with the dlpack data.

    """
    default_ws = workspace.get_workspace()
    impl = default_ws.create_tensor(scope="DLPack").FromDLPack(dlpack)
    return Tensor(impl=impl, deleter=default_ws._handle_pool)


def to_dlpack(tensor):
    """Return a dlpack tensor sharing the data.

    Parameters
    ----------
    tensor : dragon.Tensor
        The tensor to provide data.

    Returns
    -------
    PyCapsule
        The dlpack tensor.

    """
    return tensor._impl.ToDLPack(tensor.device.to_proto())
