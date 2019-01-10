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

"""The Implementation of the mpi layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon
from ..layer import Layer


class MPIBroadcastLayer(Layer):
    """The implementation of ``MPIBroadcastLayer``.

    Parameters
    ----------
    root : int
        The world rank of root. Refer `MPIParameter.root`_.

    """
    def __init__(self, LayerParameter):
        super(MPIBroadcastLayer, self).__init__(LayerParameter)
        self.arguments = {'root': LayerParameter.mpi_param.root}

    def LayerSetup(self, bottom):
        return dragon.ops.MPIBroadcast(bottom, **self.arguments)


class MPIGatherLayer(Layer):
    """The implementation of ``MPIGatherLayer``.

    Parameters
    ----------
    root : int
        The world rank of root. Refer `MPIParameter.root`_.

    """
    def __init__(self, LayerParameter):
        super(MPIGatherLayer, self).__init__(LayerParameter)
        self.arguments = {
            'root': LayerParameter.mpi_param.root,
            'num_outputs': len(self._top),
        }

    def LayerSetup(self, bottom):
        return dragon.ops.MPIGather(bottom, **self.arguments)