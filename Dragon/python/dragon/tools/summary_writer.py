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

import os

from dragon.core.tensor import Tensor
import dragon.core.workspace as ws


class ScalarSummary(object):
    """Write scalar summary.

    Examples
    --------
    >>> sw = ScalarSummary(log_dir='logs')
    >>> sw.add_summary(('loss', 2.333), 0)

    """
    def __init__(self, log_dir='logs'):
        """Construct a ScalarSummary writer.

        Parameters
        ----------
        log_dir : str
            The root folder of logs.

        Returns
        -------
        ScalarSummary
            The scalar writer.

        """
        self.log_dir = os.path.join(log_dir, 'scalar')
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

    def add_summary(self, scalar, global_step):
        """Add a summary.

        Parameters
        ----------
        scalar : tuple or Tensor
            The scalar.
        global_step : int
            The time step of this summary.

        Returns
        -------
        None

        """
        if isinstance(scalar, Tensor):
            key, value = scalar.name, ws.FetchTensor(scalar)[0]
        elif isinstance(scalar, tuple): key, value = scalar
        else: raise TypeError()
        key = key.replace('/', '_')

        with open(os.path.join(self.log_dir, key + '.txt'), 'a') as f:
            f.write(str(global_step) + ' ' + str(value) + '\n')