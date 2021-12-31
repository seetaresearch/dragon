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
"""CuDNN backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from dragon.core.device import cuda
from dragon.core.framework import sysconfig


class CuDNNModule(object):
    """CuDNN module class."""

    def __init__(self):
        self._enabled = True
        self._benchmark = False
        self._deterministic = False
        self._allow_tf32 = False

    @property
    def allow_tf32(self):
        """The flag that allows cuDNN TF32 math type or not."""
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value):
        self._allow_tf32 = value
        cuda.set_cudnn_flags(allow_tf32=value)

    @property
    def benchmark(self):
        """The flag that benchmarks fastest cuDNN algorithms or not."""
        return self._benchmark

    @benchmark.setter
    def benchmark(self, value):
        self._benchmark = value
        cuda.set_cudnn_flags(benchmark=value)

    @property
    def deterministic(self):
        """The flag that selects deterministic cuDNN algorithms or not."""
        return self._deterministic

    @deterministic.setter
    def deterministic(self, value):
        self._deterministic = value
        cuda.set_cudnn_flags(deterministic=value)

    @property
    def enabled(self):
        """The flag that uses cuDNN or not."""
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        cuda.set_cudnn_flags(enabled=value)

    @staticmethod
    def is_available():
        """Return a bool reporting if cuDNN is available.

        Returns
        -------
        bool
            ``True`` if available otherwise ``False``.

        """
        return 'cudnn_version' in sysconfig.get_build_info()

    @staticmethod
    def version():
        """Return the cuDNN version.

        Returns
        -------
        int
            The version number.

        """
        version = sysconfig.get_build_info().get('cudnn_version', None)
        if version is not None:
            major, minor, patch = [int(x) for x in version.split('.')]
            version = major * 1000 + minor * 100 + patch
        return version


# Module instances.
sys.modules[__name__] = CuDNNModule()

# Annotations.
allow_tf32: bool
benchmark: bool
deterministic: bool
enabled: bool
is_available: callable
version: callable
