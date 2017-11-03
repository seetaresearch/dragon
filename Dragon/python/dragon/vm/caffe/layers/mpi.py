# --------------------------------------------------------
# Caffe @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.ops import MPIBroadcast, MPIGather

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
        param = LayerParameter.mpi_param
        self._param = {'root': param.root}

    def Setup(self, bottom):
        super(MPIBroadcastLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return MPIBroadcast(input, **self._param)


class MPIGatherLayer(Layer):
    """The implementation of ``MPIGatherLayer``.

    Parameters
    ----------
    root : int
        The world rank of root. Refer `MPIParameter.root`_.

    """
    def __init__(self, LayerParameter):
        super(MPIGatherLayer, self).__init__(LayerParameter)
        param = LayerParameter.mpi_param
        self._param = {'root': param.root}

    def Setup(self, bottom):
        super(MPIGatherLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return MPIGather(input, nout=len(self._top), **self._param)