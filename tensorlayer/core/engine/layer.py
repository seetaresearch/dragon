# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import context
from dragon.core.util import nest
from dragon.vm.tensorlayer.core import activations
from dragon.vm.tensorlayer.core.engine import module
from dragon.vm.tensorlayer.core.engine import node


class Layer(module.Module):
    """The base layer abstraction of a neural network.

    It should be subclassed when implementing new types of layers:

    ```python
    class MyLayer(tl.layers.Layer):
        def __init__(name=None, act=None):
            super(MyLayer, self).__init__(name=name, act=act)
    ```

    """

    def __init__(self, name=None, act=None, *args, **kwargs):
        """Create a new ``Layer``.

        Parameters
        ----------
        name : str, optional.
            The optional layer name.
        act : str or function, optional
            The optional activation.

        """
        super(Layer, self).__init__(name=name)
        self._built = False
        self._nodes = []
        self._nodes_fixed = False
        self.act = activations.get(act)

    @property
    def all_weights(self):
        """Return all the weights, both trainable and non-trainable.

        Returns
        -------
        Sequence[dragon.Tensor]
            The weights sequence.

        """
        return self.trainable_weights + self.nontrainable_weights

    @property
    def name(self):
        """Return the layer name.

        Returns
        -------
        str
            The layer name.

        """
        return super(Layer, self).name

    @property
    def nontrainable_weights(self):
        """Return the non-trainable weights.

        Returns
        -------
        Sequence[dragon.Tensor]
            The weights sequence.

        """
        return self._nontrainable_weights

    @property
    def trainable_weights(self):
        """Return the trainable weights.

        Returns
        -------
        Sequence[dragon.Tensor]
            The weights sequence.

        """
        return self._trainable_weights

    @module.Module.training.setter
    def training(self, mode):
        """Set the training mode.

        Parameters
        ----------
        mode : bool
            **True** for training otherwise evaluation.

        """
        self._training = mode

    def build(self, input_shapes):
        """Method to define the weights.

        Parameters
        ----------
        input_shapes : Sequence[Sequence[int]]
            The shape of inputs.

        """
        self._built = True

    def forward(self, inputs, **kwargs):
        """Method to define the forward operations.

        Parameters
        ----------
        inputs : Sequence[dragon.Tensor]
            The inputs.

        Returns
        -------
        Sequence[dragon.Tensor]
            The outputs.

        """
        pass

    def _add_node(self, inputs, outputs):
        """Add a layer node for inputs and outputs.

        Parameters
        ----------
        inputs : Sequence[dragon.Tensor]
            The input tensors.
        outputs : Sequence[dragon.Tensor]
            The output tensors.

        """
        inputs = nest.flatten(inputs)
        outputs = nest.flatten(outputs)
        input_info = [getattr(e, '_info', [None, None]) for e in inputs]

        self._nodes.append(
            node.LayerNode(
                self,
                node_index=len(self._nodes),
                in_nodes=[e[0] for e in input_info],
                in_tensor_idxes=[e[1] for e in input_info],
                in_tensors=inputs,
                out_tensors=outputs,
            )
        )

        for idx, tensor in enumerate(outputs):
            tensor._info = (self._nodes[-1], idx)

    def _fix_nodes(self):
        """Fix layer nodes to stop growing."""
        self._nodes_fixed = True

    def __call__(self, inputs, **kwargs):
        """The preprocessor for ``self.forward(...)``."""
        with context.name_scope(self.name):
            # Maybe build the layer at the first time.
            if not self._built:
                input_list = nest.flatten(inputs)
                input_shapes = None
                if all(hasattr(x, 'shape') for x in input_list):
                    input_shapes = [x.shape for x in input_list]
                    if not nest.is_sequence(inputs):
                        input_shapes = input_shapes[0]
                self.build(input_shapes)

            # Call the forward implementation to get outputs.
            outputs = self.forward(inputs, **kwargs)

        # Record the nodes if necessary.
        if not self._nodes_fixed:
            self._add_node(inputs, outputs)

        return outputs

    def __delitem__(self, key):
        raise TypeError('The Layer API does not allow to use the method: `__delitem__`')

    def __repr__(self):
        return 'Layer'

    def __setitem__(self, key, item):
        raise TypeError('The Layer API does not allow to use the method: `__setitem__`')
