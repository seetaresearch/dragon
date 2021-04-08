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
"""Autograph configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def set_scheduler(scheduler='SIMPLE'):
    """Set the scheduler for symbolic graph.

    Parameters
    ----------
    scheduler : {'SIMPLE', 'FUSION'}, optional
        The scheduler type.

    """
    if scheduler not in ('SIMPLE', 'FUSION'):
        raise ValueError('Unsupported scheduler: ' + scheduler)
    if scheduler == 'SIMPLE':
        config.config().graph_type = ''
    elif scheduler == 'FUSION':
        config.config().graph_type = 'FusionGraph'


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
