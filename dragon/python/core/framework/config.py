# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

"""Define the global configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading


class Config(object):
    """Store the common configurations for frontend."""

    def __init__(self):
        # The type of device.
        # Enumeration in ('cpu', 'cuda', 'cnml').
        self.device_type = 'cpu'
        # The device index.
        self.device_index = 0
        # The global random seed.
        self.random_seed = 3

        # The graph type for various scheduling.
        self.graph_type = ''
        # The graph optimization level.
        self.graph_optimization = 3
        # The graph verbosity level.
        self.graph_verbosity = 0
        # The execution mode for graph.
        self.graph_execution = 'GRAPH_MODE'

        # The directory to store logging files.
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
