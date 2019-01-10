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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon.config as config


DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2**31 - 1


def _truncate_seed(seed):
  return seed % _MAXINT32  # Truncate to fit into 32-bit integer


def get_seed(op_seed):
    """Return the global random seed.

    Parameters
    ----------
    op_seed : int
        The optional seed to use.

    Return
    ------
    tuple
        A tuple of two ints for using.

    """
    graph_seed = config.GetRandomSeed()
    if graph_seed is not None:
        if op_seed is None:
            # pylint: disable=protected-access
            op_seed = graph_seed
        seeds = _truncate_seed(graph_seed), _truncate_seed(op_seed)
    else:
        if op_seed is not None:
            seeds = DEFAULT_GRAPH_SEED, _truncate_seed(op_seed)
        else:
            seeds = None, None
    return seeds


def set_random_seed(seed):
    """Set the global random seed.

    Parameters
    ----------
    seed : int
        The seed to use.

    """
    config.SetRandomSeed(seed)

