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
"""System configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from dragon.core.framework import backend


def get_build_info():
    """Return the environment information of built binaries.

    The return value is a dictionary with string keys:

    * cpu_features
    * cuda_version
    * cudnn_version
    * mps_version
    * is_cuda_build
    * is_mps_build
    * third_party

    Returns
    -------
    dict
        The info dict.

    """
    build_info = collections.OrderedDict()
    build_info_str = backend.GetBuildInformation()
    for entry in build_info_str.split('\n'):
        k, v = entry.split(':')
        if len(v) > 0:
            build_info[k] = v[1:]
    build_info['is_cuda_build'] = 'cuda_version' in build_info
    build_info['is_mps_build'] = 'mps_version' in build_info
    return build_info


def get_include():
    """Return the directory of framework header files.

    Returns
    -------
    str
        The include directory.

    """
    core_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(os.path.dirname(core_root), 'include')


def get_lib():
    """Return the directory of framework libraries.

    Returns
    -------
    str
        The library directory.

    """
    core_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(os.path.dirname(core_root), 'lib')
