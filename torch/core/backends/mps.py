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
"""MPS backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.device import mps
from dragon.core.framework import sysconfig


def is_available():
    """Return a bool reporting if MPS is available.

    Returns
    -------
    bool
        ``True`` if available otherwise ``False``.

    """
    return mps.is_available()


def is_built():
    """Return a bool reporting if built with MPS support.

    Returns
    -------
    bool
        ``True`` if built otherwise ``False``.

    """
    return sysconfig.get_build_info()['is_mps_build']
