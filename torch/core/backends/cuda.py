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
"""CUDA backend."""

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
    return sysconfig.get_build_info()["is_cuda_build"]


# Module instances.
matmul = CuBLASModule()
