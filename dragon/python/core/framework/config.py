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
"""Framework configurations."""

import threading

from dragon.core.framework import backend


class Config(object):
    """Framework configuration class."""

    def __init__(self):
        # Device type.
        # Enumeration in ('cpu', 'cuda', 'mps', 'mlu').
        self.device_type = "cpu"
        # Device index.
        self.device_index = 0
        # Graph type for various scheduling.
        self.graph_type = ""
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
    """Set the random seed for cpu device.

    Parameters
    ----------
    seed : int
        The seed to use.

    """
    backend.SetRandomSeed(seed)


_config = None
_config_lock = threading.Lock()


def _create_config():
    global _config
    with _config_lock:
        if _config is None:
            _config = Config()
