# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import string
from dragon.vm.tensorlayer.core.engine import module


class LayerList(module.Module):
    """The sequential layer to stack a group of layers."""

    def __init__(self, layers, name=None):
        """Create a new ``LayerList``.

        Parameters
        ----------
        layers : Sequence[dragon.vm.tensorlayer.layers.Layer]
            The layers to stack.
        name : str, optional
            The optional layer name.

        """
        super(LayerList, self).__init__(name=name)
        self._layers = layers
        for index, layer in enumerate(self._layers):
            if layer._name is None:
                layer._name = str(index)

    @property
    def all_weights(self):
        """Return all the weights, both trainable and non-trainable.

        Returns
        -------
        Sequence[dragon.Tensor]
            The weights sequence.

        """
        return self.weights

    @property
    def name(self):
        """Return the layer name.

        Returns
        -------
        str
            The layer name.

        """
        return super(LayerList, self).name

    def forward(self, inputs, *args, **kwargs):
        """Forward the computation sequentially.

        Parameters
        ----------
        inputs : Sequence[dragon.Tensor]
            The inputs.

        Returns
        -------
        Sequence[dragon.Tensor]
            The outputs.

        """
        outputs = inputs
        for layer in self._layers:
            outputs = layer.forward(outputs, *args, **kwargs)
        return outputs

    def __getitem__(self, idx):
        """Return the specified layer selected by index.

        Parameters
        ----------
        idx : int
            The layer index in the sequence.

        Returns
        -------
        dragon.vm.tensorlayer.layers.Layer
            The layer instance.

        """
        if isinstance(idx, slice):
            return LayerList(list(self._layers)[idx])
        else:
            return self._layers[idx]

    def __len__(self):
        """Return the number of sequential layers.

        Returns
        -------
        int
            The number of layers.

        """
        return len(self._layers)

    def __repr__(self):
        tmp_str = 'LayerList' + '(\n'
        for idx, layer in enumerate(self._layers):
            mod_str = layer.__repr__()
            mod_str = string.add_indent(mod_str, 2)
            tmp_str = tmp_str + '  (' + str(idx) + '): ' + mod_str + '\n'
        tmp_str = tmp_str + ')'
        return tmp_str
