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
"""Autograph configurations."""

from dragon.core.framework import config


def set_optimization(level=1):
    """Set the optimization for graph ir.

    Following levels are defined (default=3):

    * level = ``0``: Do nothing.

    * level = ``1``: Eliminate the unused outputs and operators.

    * level = ``2``: Apply the inplace to inputs if available.

    * level = ``3``: Allocate the shared buffer to outputs if available.

    Parameters
    ----------
    level : int, optional, default=3
        The optimization level.

    """
    config.config().graph_optimization = level


def set_scheduler(scheduler="SIMPLE"):
    """Set the scheduler for symbolic graph.

    Parameters
    ----------
    scheduler : {'SIMPLE', 'FUSION'}, optional
        The scheduler type.

    """
    if scheduler not in ("SIMPLE", "FUSION"):
        raise ValueError("Unsupported scheduler: " + scheduler)
    if scheduler == "SIMPLE":
        config.config().graph_type = ""
    elif scheduler == "FUSION":
        config.config().graph_type = "FusionGraph"


def set_verbosity(level=1):
    """Set the verbosity for graph ir.

    Following levels are defined (default=0):

    * level = ``0``: Do nothing.

    * level = ``1``: Print the optimized GraphIR.

    * level = ``2``: Print the raw GraphIR.

    Parameters
    ----------
    level : int, optional, default=1
        The verbosity level.

    """
    config.config().graph_verbosity = level
