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
"""CNNL backend."""

import sys

from dragon.core.device import mlu
from dragon.core.framework import sysconfig


class CNNLModule(object):
    """CNNL module class."""

    def __init__(self):
        self._enabled = True

    @property
    def enabled(self):
        """The flag that uses CNNL or not."""
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        mlu.set_cnnl_flags(enabled=value)

    @staticmethod
    def is_available():
        """Return a bool reporting if CNNL is available.

        Returns
        -------
        bool
            ``True`` if available otherwise ``False``.

        """
        return "cnnl_version" in sysconfig.get_build_info()

    @staticmethod
    def version():
        """Return the CNNL version.

        Returns
        -------
        int
            The version number.

        """
        version = sysconfig.get_build_info().get("cnnl_version", None)
        if version is not None:
            major, minor, patch = [int(x) for x in version.split(".")]
            version = major * 1000 + minor * 100 + patch
        return version


# Module instances.
sys.modules[__name__] = CNNLModule()

# Annotations.
enabled: bool
is_available: callable
version: callable
