# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from layer import Layer
from dragon.ops import MPIBroadcast, MPIGather

class MPIBroadcastLayer(Layer):
    def __init__(self, LayerParameter):
        super(MPIBroadcastLayer, self).__init__(LayerParameter)
        param = LayerParameter.mpi_param
        self._param = {'root': param.root}

    def Setup(self, bottom):
        super(MPIBroadcastLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return MPIBroadcast(input, **self._param)

class MPIGatherLayer(Layer):
    def __init__(self, LayerParameter):
        super(MPIGatherLayer, self).__init__(LayerParameter)
        param = LayerParameter.mpi_param
        self._param = {'root': param.root}

    def Setup(self, bottom):
        super(MPIGatherLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return MPIGather(input, nout=len(self._top), **self._param)