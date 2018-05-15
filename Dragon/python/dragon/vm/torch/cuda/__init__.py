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

import dragon as dg


def is_available():
    """Returns a bool indicating if CUDA is currently available.

    Returns
    -------
    boolean
        ``True`` if your device(s) support CUDA otherwise ``False``.

    """
    return dg.config.IsCUDADriverSufficient()


def set_device(device):
    """Sets the current device.

    Parameters
    ----------
    device : int
        The id of device to use.

    """
    if device >= 0: dg.SetGPU(device)
