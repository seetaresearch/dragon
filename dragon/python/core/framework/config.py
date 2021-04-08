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
"""Framework configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from dragon.core.framework import backend


class Config(object):
    """Framework configuration class."""

    def __init__(self):
        # Device type.
        # Enumeration in ('cpu', 'cuda', 'cnml').
        self.device_type = 'cpu'
        # Device index.
        self.device_index = 0
        # Device random seed.
        self.random_seed = 3
        # Graph type for various scheduling.
        self.graph_type = ''
        # Graph optimization level.
        self.graph_optimization = 3
        # Graph verbosity level.
        self.graph_verbosity = 0
        # Directory to store logging files.
        self.log_dir = None


def config():
    """Return a singleton config object.

    Returns
    -------
    Config
        The config object.

    """
    if _config is None:
        _create_config()
    return _config


def get_num_threads():
    """Return the number of threads for cpu parallelism.

    Returns
    -------
    num : int
        The number of threads to use.

    """
    return backend.GetNumThreads()


def set_num_threads(num):
    """Set the number of threads for cpu parallelism.

    Parameters
    ----------
    num : int
        The number of threads to use.

    """
    backend.SetNumThreads(num)


def set_random_seed(seed):
    """Set the global random seed.

    Parameters
    ----------
    seed : int
        The seed to use.

    """
    config().random_seed = seed


_config = None
_config_lock = threading.Lock()


def _create_config():
    global _config
    with _config_lock:
        if _config is None:
            _config = Config()
