# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

"""Implementation for the ``Layer`` C++ class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph.tensor import RefTensor
from dragon.core.eager import context as eager_context
from dragon.core.framework import context


class Layer(object):
    """The abstraction of ``caffe.Layer``."""

    def __init__(self, layer_param):
        """Create a ``Layer``.

        Parameters
        ----------
        layer_param : LayerParameter
            The parameter containing layer arguments.

        """
        self._proto = layer_param
        self._name = layer_param.name
        self._arguments, self.arguments = {'name': self._name}, {}

        # Store the inputs, outputs and trainable parameters.
        self._bottom, self._top, self._blobs = [], [], []
        for blob in layer_param.bottom:
            self._bottom.append(blob)
        for blob in layer_param.top:
            self._top.append(blob)

        # Store the loss weight to apply gradients.
        self._loss_weight = layer_param.loss_weight \
            if len(layer_param.loss_weight) > 0 else None

        # Optional mirror stage argument for memory optimization.
        if layer_param.HasField('mirror_stage'):
            self._arguments['mirror_stage'] = layer_param.mirror_stage

    @property
    def bottom(self):
        """Return the bottom names."""
        return self._bottom

    @property
    def loss_weight(self):
        """Return the loss weight."""
        return self._loss_weight

    @property
    def top(self):
        """Return the top names."""
        return self._top

    def add_blob(self, value=None, filler=None, no_grad=False):
        """Add a weight blob into this layer."""
        # Use a fixed name in the current workspace.
        # Note that a non-empty tensor scope will make it
        # impossible to load/save models. You should use
        # a new workspace instead of the terrible name scope.
        scoped_name = context.get_name_scope() + self._name
        param_name = scoped_name + '/param:{}'.format(len(self._blobs))

        # Set the name explicitly.
        variable = RefTensor(param_name)
        variable_grad = RefTensor(param_name + '_grad')

        if filler is not None:
            variable._register_as(**filler)
        else:
            # Register a constant filler by default.
            value = value if value else 0
            variable.constant(value=value)

        # Determine whether to disable the gradients explicitly.
        if no_grad is True:
            variable_grad = None

        # Append to the blobs.
        self._blobs.append({'data': variable, 'diff': variable_grad})

    def setup(self, bottom):
        # Merge the arguments, then setup up the specific layer.
        self.arguments = dict(self.arguments, **self._arguments)
        bottom = bottom[0] if len(bottom) == 1 else bottom
        with eager_context.graph_mode():
            return self.__call__(bottom)

    @classmethod
    def get_filler(cls, layer_param, filler_name):
        """Construct a filler from the parameter."""
        if layer_param.HasField(filler_name):
            filler = getattr(layer_param, filler_name)
            return {
                'type': filler.type.lower(),
                'value': filler.value,
                'low': filler.min,
                'high': filler.max,
                'mean': filler.mean,
                'std': filler.std,
            }
        return None

    def __call__(self, bottom):
        """Define the forward pipeline."""
        raise NotImplementedError
