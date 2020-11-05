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
"""List the exported C++ API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import ctypes
import os
import platform
import sys

if platform.system() == 'Windows':
    package_root = os.path.dirname(__file__)
    dll_paths = [os.path.join(package_root, 'lib')]
    dll_paths = list(filter(os.path.exists, dll_paths)) + [os.environ['PATH']]
    os.environ['PATH'] = ';'.join(dll_paths)

try:
    from dragon.lib.libdragon_python import *
except ImportError as e:
    print('Cannot import dragon. Error: {0}'.format(str(e)))
    sys.exit(1)


@contextlib.contextmanager
def dlopen_guard(extra_flags=ctypes.RTLD_GLOBAL):
    """
    Context manager to open_guardflags ().

    Args:
        extra_flags: (todo): write your description
        ctypes: (todo): write your description
        RTLD_GLOBAL: (todo): write your description
    """
    old_flags = None
    if _getdlopenflags and _setdlopenflags:
        old_flags = _getdlopenflags()
        _setdlopenflags(old_flags | extra_flags)
    try:
        yield
    finally:
        if old_flags is not None:
            _setdlopenflags(old_flags)


def load_library(library_location):
    """Load a shared library.

    The library should contain objects registered
    in a registry, e.g., ``CPUOperatorRegistry``.

    Parameters
    ----------
    library_location : str
        The path of the shared library file.

    """
    if not os.path.exists(library_location):
        raise FileNotFoundError('Invalid path: %s' % library_location)
    with dlopen_guard():
        ctypes.cdll.LoadLibrary(library_location)
    _reload_registry()


def _reload_registry():
    """Reload the list of registries."""
    global REGISTERED_OPERATORS
    global NO_GRADIENT_OPERATORS
    REGISTERED_OPERATORS = frozenset(s for s in RegisteredOperators())
    NO_GRADIENT_OPERATORS = frozenset(s for s in NoGradientOperators())


_getdlopenflags = getattr(sys, 'getdlopenflags', None)
_setdlopenflags = getattr(sys, 'setdlopenflags', None)

REGISTERED_OPERATORS = frozenset(s for s in RegisteredOperators())
NO_GRADIENT_OPERATORS = frozenset(s for s in NoGradientOperators())
