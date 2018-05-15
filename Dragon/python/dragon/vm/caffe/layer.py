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

from dragon.core.tensor import Tensor

from .utils import ToFillerArgs


class Layer(object):
    """
    Layer is the basic structure for parsing text format definition.

    We further extent it with MPI and memory optimization utilities.
    """
    def __init__(self, LayerParameter):
        """Construct a Layer.

        Parameters
        ----------
        LayerParameter : caffe_pb2.LayerParameter
            The parameter of ``Layer``.

        Returns
        -------
        Layer
            The layer.

        """
        self._bottom = []; self._top = []

        for bottom in LayerParameter.bottom:
            self._bottom.append(bottom)

        for top in LayerParameter.top:
            self._top.append(top)

        self._name = LayerParameter.name
        self._blobs = []
        self._param = {}
        self._common_param = {}

        self._loss_weight = None if len(LayerParameter.loss_weight) == 0 \
                                 else LayerParameter.loss_weight

        for include in LayerParameter.include:
            mpi_rank = [int(rank) for rank in include.mpi_rank]
            if len(mpi_rank) > 0: self._common_param['mpi_ranks'] = mpi_rank

        if LayerParameter.HasField('mirror_stage'):
            self._common_param['mirror_stage'] = LayerParameter.mirror_stage

    def Setup(self, bottom):
        """Setup the parameters.

        Parameters
        ----------
        bottom : list of Tensor
            The inputs.

        Returns
        -------
        None

        References
        ---------=
        The implementation of `LayerSetUp(layer.hpp, L91)`_.

        """
        self._param = dict(self._param, **self._common_param)

    def Fill(self, tensor, layer_param, filler):
        """Register the fillers.

        Parameters
        ----------
        tensor : Tensor
            The tensor to register.
        layer_param : caffe_pb2.LayerParameter.XXXParameter
            The parameter of specific ``XXXLayer``.
        filler : str
            The name of filler.

        Returns
        -------
        None

        Examples
        --------
        >>> from dragon.core.tensor import Tensor
        >>> weight = Tensor().Variable()
        >>> conv_param = LayerParameter.convolution_param
        >>> Fill(weight, conv_param, 'weight_filler')
        >>> Fill(weight, conv_param, 'bias_filler')

        """
        if layer_param.HasField(filler):
            filler = getattr(layer_param, filler)
            tensor.Fill(filler.type, **ToFillerArgs(filler))
        else: tensor.Fill('constant')
