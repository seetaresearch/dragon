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
"""CUDA random utilities."""

from dragon.core.framework import backend


def manual_seed(seed, device_index=None):
    """Set the random seed for cuda device.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    seed : int
        The seed to use.
    device_index : int, optional
        The device index.

    """
    device_index = -1 if device_index is None else device_index
    backend.cudaSetRandomSeed(device_index, int(seed))


def manual_seed_all(seed):
    """Set the random seed for all cuda devices.

    Parameters
    ----------
    seed : int
        The seed to use.

    """
    for i in range(backend.cudaGetDeviceCount()):
        backend.cudaSetRandomSeed(i, int(seed))
