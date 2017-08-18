# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.vm.caffe.utils import ToFillerArgs

class Layer(object):
    def __init__(self, LayerParameter):
        self._bottom = []; self._top = []

        for bottom in LayerParameter.bottom:
            self._bottom.append(bottom)

        for top in LayerParameter.top:
            self._top.append(top)

        self._name = LayerParameter.name
        self._blobs = []
        self._param = {}
        self._common_param = {}

        for include in LayerParameter.include:
            mpi_rank = [int(rank) for rank in include.mpi_rank]
            if len(mpi_rank) > 0: self._common_param['mpi_rank'] = mpi_rank

        if LayerParameter.HasField('mirrow_stage'):
            self._common_param['mirrow_stage'] = LayerParameter.mirrow_stage

    def Setup(self, bottom):
        self._param = dict(self._param, **self._common_param)

    def Fill(self, tensor, param, filler):
        """ wrapper for caffe filler """
        if param.HasField(filler):
            filler = getattr(param, filler)
            tensor.Fill(filler.type, **ToFillerArgs(filler))
        else: tensor.Fill('constant')
