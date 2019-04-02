# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

"""Implementation the singleton utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import config as _cfg


_GLOBAL_CAFFE_ROOT_SOLVER = True


def set_mode_cpu():
    """Set to the CPU mode. [**PyCaffe Style**]

    Returns
    -------
    None

    References
    ----------
    The implementation of `set_mode_cpu(_caffe.cpp, L51)`_.

    """
    _cfg.EnableCPU()


def set_mode_gpu():
    """Set to the GPU mode. [**PyCaffe Style**]

    Returns
    -------
    None

    References
    ----------
    The implementation of `set_mode_gpu(_caffe.cpp, L52)`_.

    """
    _cfg.EnableCUDA()


def set_device(device):
    """Set the active device. [**PyCaffe Style**]

    Returns
    -------
    None

    References
    ----------
    The implementation of `SetDevice(common.cpp, L65)`_.

    """
    _cfg.SetGPU(device)


def set_random_seed(seed):
    """Set the global random seed. [**PyCaffe Style**]

    Parameters
    ----------
    seed : int
        The random seed.

    Returns
    -------
    None

    References
    ----------
    The implementation of `set_random_seed(_caffe.cpp, L71)`_.

    """
    _cfg.SetRandomSeed(seed)


def root_solver():
    """Whether this node is root.

    Returns
    -------
    boolean
        True, if setting it before.

    References
    ----------
    The implementation of `root_solver(common.hpp, L164)`_.

    """
    return _GLOBAL_CAFFE_ROOT_SOLVER


def set_root_solver(val):
    """Set this node to the root.

    Parameters
    ----------
    val : boolean
        Whether to become the root.

    References
    ----------
    The implementation of `set_root_solver(common.hpp, L165)`_.

    """
    global _GLOBAL_CAFFE_ROOT_SOLVER
    _GLOBAL_CAFFE_ROOT_SOLVER = val