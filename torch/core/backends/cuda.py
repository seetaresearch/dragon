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
"""CUDA backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.device import cuda
from dragon.core.framework import sysconfig


class CuBLASModule(object):
    """CuBLAS module class."""

    def __init__(self):
        self._allow_tf32 = False

    @property
    def allow_tf32(self):
        """The flag that allows cuBLAS TF32 math type or not."""
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value):
        self._allow_tf32 = value
        cuda.set_cublas_flags(allow_tf32=value)


def is_built():
    """Return a bool reporting if built with CUDA support.

    Returns
    -------
    bool
        ``True`` if built otherwise ``False``.

    """
    return sysconfig.get_build_info()['is_cuda_build']


# Module instances.
matmul = CuBLASModule()
