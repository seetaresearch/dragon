from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import context
from dragon.vm.tensorlayer.core.engine import module


class Model(module.Module):
    """Compose a group of layers with training features."""

    def __init__(self, name=None):
        """Create a ``Model`` instance.

        Parameters
        ----------
        name : str, optional
            The optional model name.

        """
        super(Model, self).__init__(name=name)
        self._config = None
        self._nodes_fixed = False

    @property
    def all_layers(self):
        """Return all the layers in this model.

        Returns
        -------
        Sequence[dragon.vm.tensorlayer.layers.Layer]
            The layer sequence.

        """
        return self.modules

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
        """Return the model name.

        Returns
        -------
        str
            The model name.

        """
        return super(Model, self).name

    def train(self):
        """Set the model in training mode."""
        self.training = True

    def eval(self):
        """Set the model in evaluation mode."""
        self.training = False

    def _fix_nodes(self):
        """Fix the layer nodes to stop growing."""
        for layer in self.all_layers:
            try:
                layer._fix_nodes()
            except AttributeError:
                pass
        self._nodes_fixed = True

    def __call__(self, inputs, **kwargs):
        """The preprocessor for ``self.forward(...)``."""
        if self._nodes_fixed is False:
            self._fix_nodes()
        with context.name_scope(self.name):
            return self.forward(inputs, **kwargs)

    def __repr__(self):
        tmpstr = self.name + '(\n'
        for idx, layer in enumerate(self.all_layers):
            modstr = layer.__repr__()
            modstr = self._addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + layer.name + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr
