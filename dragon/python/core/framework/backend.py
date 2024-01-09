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
"""Framework backend."""

import contextlib
import ctypes
import os
import platform
import sys

_get_dlopen_flags = getattr(sys, "getdlopenflags", None)
_set_dlopen_flags = getattr(sys, "setdlopenflags", None)

if platform.system() == "Windows":
    package_root = os.path.abspath(os.path.dirname(__file__) + "/../..")
    dll_paths = [os.path.join(package_root, "lib")]
    dll_paths = list(filter(os.path.exists, dll_paths)) + [os.environ["PATH"]]
    os.environ["PATH"] = ";".join(dll_paths)

try:
    from dragon.lib.libdragon_python import *  # noqa
except ImportError as e:
    print("Cannot import dragon. Error: {0}".format(str(e)))
    sys.exit(1)


@contextlib.contextmanager
def dlopen_guard(extra_flags=ctypes.RTLD_GLOBAL):
    old_flags = None
    if _get_dlopen_flags and _set_dlopen_flags:
        old_flags = _get_dlopen_flags()
        _set_dlopen_flags(old_flags | extra_flags)
    try:
        yield
    finally:
        if old_flags is not None:
            _set_dlopen_flags(old_flags)


def load_library(library_location):
    """Load a shared library.

    The library should contain objects registered
    in a registry.py, e.g., ``CPUOperatorRegistry``.

    Parameters
    ----------
    library_location : str
        The path of the shared library file.

    """
    if not os.path.exists(library_location):
        raise FileNotFoundError("Invalid path: %s" % library_location)
    with dlopen_guard():
        ctypes.cdll.LoadLibrary(library_location)
