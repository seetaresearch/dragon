# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/pytorch/pytorch/blob/master/torch/cuda/__init__.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon


def is_available():
    """Return a bool indicating if CUDA is currently available.

    Returns
    -------
    boolean
        ``True`` if your device(s) support CUDA otherwise ``False``.

    """
    return dragon.cuda.IsCUDADriverSufficient()


def current_device():
    """Return the index of the current active device.
    
    Returns
    -------
    int
        The index of device.

    """
    return dragon.cuda.GetDevice()


def synchronize():
    """Waits for all kernels in the default stream on current device.

    Returns
    -------
    None

    """
    return dragon.cuda.SynchronizeStream()