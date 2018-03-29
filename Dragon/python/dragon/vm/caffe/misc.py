# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

import dragon.config as config

_root_solver = True

__all__ = [
    'set_mode_cpu',
    'set_mode_gpu',
    'set_device',
    'set_random_seed',
    'root_solver',
    'set_root_solver'
]


def set_mode_cpu():
    """Set to the CPU mode. [**PyCaffe Style**]

    Returns
    -------
    None

    References
    ----------
    The implementation of `set_mode_cpu(_caffe.cpp, L51)`_.

    """
    config.EnableCPU()


def set_mode_gpu():
    """Set to the GPU mode. [**PyCaffe Style**]

    Returns
    -------
    None

    References
    ----------
    The implementation of `set_mode_gpu(_caffe.cpp, L52)`_.

    """
    config.EnableCUDA()


def set_device(device):
    """Set the active device. [**PyCaffe Style**]

    Returns
    -------
    None

    References
    ----------
    The implementation of `SetDevice(common.cpp, L65)`_.

    """
    config.SetGPU(device)


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
    config.SetRandomSeed(seed)


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
    global _root_solver
    return _root_solver


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
    global _root_solver
    _root_solver = val