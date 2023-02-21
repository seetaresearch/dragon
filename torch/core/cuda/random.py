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
"""CUDA random utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
