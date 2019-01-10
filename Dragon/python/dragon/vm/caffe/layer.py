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

"""The implementation of the ``Layer`` C++ class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon


class Layer(object):
    """Layer is the basic structure for parsing text format definition.

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
        self._proto = LayerParameter
        self._name = LayerParameter.name
        self._arguments, self.arguments = {'name': self._name}, {}

        # Store the inputs, outputs and trainable parameters
        self._bottom, self._top, self._blobs = [], [], []
        for bottom in LayerParameter.bottom: self._bottom.append(bottom)
        for top in LayerParameter.top: self._top.append(top)

        # Store the loss weight to apply gradients
        self._loss_weight = LayerParameter.loss_weight \
            if len(LayerParameter.loss_weight) > 0 else None

        # Optional include rule for MPI Layer
        for include in LayerParameter.include:
            mpi_rank = [int(rank) for rank in include.mpi_rank]
            if len(mpi_rank) > 0: self._arguments['mpi_ranks'] = mpi_rank

        # Optional mirror stage argument for memory optimization
        if LayerParameter.HasField('mirror_stage'):
            self._arguments['mirror_stage'] = LayerParameter.mirror_stage

    def LayerSetup(self, bottom):
        # Implemented by the specific layer
        raise NotImplementedError()

    def Setup(self, bottom):
        # Merge the arguments, then setup up the specific layer
        self.arguments = dict(self.arguments, **self._arguments)
        return self.LayerSetup(bottom[0] if len(bottom) == 1 else bottom)

    def AddBlob(self, value=None, filler=None, enforce_no_grad=None):
        # Use a a fixed name in the current workspace
        # Note that a non-empty tensor scope will make it
        # impossible to load/save caffe models. You should use
        # a new workspace instead of the terrible name scope
        scoped_name = dragon.get_default_name_scope() + self._name
        param_name = scoped_name + '/param:{}'.format(len(self._blobs))

        # Set the name explicitly
        variable = dragon.Tensor.Ref(param_name)
        variable_grad = dragon.Tensor.Ref(param_name + '_grad')

        if filler is not None:
            variable.Fill(**filler)
        else:
            # Register a constant filler by default
            value = value if value else 0
            variable.Constant(value=value)

        # Determine whether we have disabled the gradients explicitly
        if enforce_no_grad is not None:
            variable_grad = None

        # Append to the blobs
        self._blobs.append({'data': variable, 'diff': variable_grad})

    def GetFiller(self, layer_param, filler_name):
        if layer_param.HasField(filler_name):
            filler = getattr(layer_param, filler_name)
            return {
                'type': filler.type.lower(), 'value': filler.value,
                'low': filler.min, 'high': filler.max,
                'mean': filler.mean, 'std': filler.std,
            }
        return None