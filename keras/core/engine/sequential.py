# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Sequential layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from dragon.core.util import inspect
from dragon.core.util import nest
from dragon.vm.keras.core.engine import base_layer


class Sequential(base_layer.Layer):
    """Stack a group of layers and run sequentially.

    Examples:

    ```python
    conv_triplet = tf.keras.Sequential([
        tf.keras.layers.Conv2D(2, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(inplace=True),
    ])
    ```

    """

    _MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain((
        '_layer_call_argspecs',
        base_layer.Layer._MODULE_IGNORED_PROPERTIES,
    )))

    def __init__(self, layers=None, name=None):
        """Create a ``Sequential`` layer.

        Parameters
        ----------
        layers : Sequence[dragon.vm.tensorflow.keras.layers.Layer]
            The layers to stack.
        name : str, optional
            The layer name.

        """
        super(Sequential, self).__init__(name=name)
        self._layers = []
        self._layer_call_argspecs = {}
        if layers is not None:
            for layer in nest.flatten(layers):
                self.add(layer)

    @property
    def layers(self):
        """Return the stacked layers.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.keras.layers.Layer]
            The layers.

        """
        return self._layers

    def add(self, layer):
        """Add a layer into the stack.

        Parameters
        ----------
        layer : dragon.vm.tensorflow.keras.layers.Layer
            The layer to add.

        """
        if not isinstance(layer, base_layer.Layer):
            raise TypeError('Excepted the <layer> should be '
                            'an instance of <tf.keras.layers.Layer>, '
                            'Got: ' + str(layer))
        self.built = False
        if layer._name is None:
            layer._name = str(len(self._layers))
        self._layers.append(layer)
        self._layer_call_argspecs[layer] = inspect.getfullargspec(layer.call)

    def call(self, inputs, training=None):
        """Call the layers sequentially."""
        outputs = inputs
        for layer in self._layers:
            kwargs = {}
            argspec = self._layer_call_argspecs[layer].args
            if 'training' in argspec:
                kwargs['training'] = training
            outputs = layer(inputs, **kwargs)
            inputs = outputs
        return outputs

    def pop(self):
        """Remove the last layer in the stack."""
        if not self._layers:
            raise TypeError('There are no layers in the model.')
        layer = self._layers.pop()
        self._layer_call_argspecs.pop(layer)
        if not self._layers:
            self.built = False
