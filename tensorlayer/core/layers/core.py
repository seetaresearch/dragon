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
from dragon.core.framework import workspace
from dragon.core.util import nest
from dragon.vm.tensorlayer.core import initializers


class LayerMetaclass(object):
    """Meta class for layer like objects."""

    def __init__(self, name=None):
        self._name = name
        self._all_weights = None
        self._trainable_weights = None
        self._nontrainable_weights = None
        self._nodes_fixed = False
        self._training = True

    @property
    def name(self):
        """Return the layer name.

        Returns
        -------
        str
            The layer name.

        """
        if self._name is None:
            self._init_set_name()
        return self._name

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
    def training(self):
        return self._training

    @training.setter
    def training(self, value):
        self._training = value

    @property
    def trainable_weights(self):
        """Return the trainable weights.

        Returns
        -------
        Sequence[dragon.Tensor]
            The weights sequence.

        """
        return self._trainable_weights

    def forward(self, inputs, **kwargs):
        """Method to define the forward operations."""
        pass

    def _init_set_name(self, name=None, zero_based=True):
        """Set the model name when necessary."""
        if name is None:
            self._name = workspace.get_dummy_name(
                basename=self.__class__.__name__.lower(),
                domain='Object',
                zero_based=zero_based,
            )
        else:
            self._name = name

    def _fix_nodes(self):
        """Fix layer nodes to stop growing."""
        self._nodes_fixed = True


class Layer(LayerMetaclass):
    """Represent a single layer of a neural network.

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
        self.act = act

    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape_mem = [t.shape for t in tensors]
        else:
            shape_mem = tensors.shape
        return shape_mem

    @property
    def all_weights(self):
        """Return all the weights, both trainable and non-trainable.

        Returns
        -------
        Sequence[dragon.Tensor]
            The weights sequence.

        """
        if self._all_weights is None:
            self._all_weights = []
            if self._trainable_weights is not None:
                self._all_weights.extend(self._trainable_weights)
            if self._nontrainable_weights is not None:
                self._all_weights.extend(self._nontrainable_weights)
        return self._all_weights

    def build(self, inputs_shape):
        """Method to define the weights."""
        self._built = True

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
            LayerNode(
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

    def _release_memory(self):
        """
        WARINING: This function should be called with great caution.

        self.inputs and self.outputs will be set as None but not deleted in order to release memory.
        """
        # FIXME : not understand why saving inputs/outputs shape
        for node in self._nodes:
            node.in_tensors = None
            node.out_tensors = None

    def _get_weights(
        self,
        name=None,
        shape=None,
        init=initializers.glorot_uniform(),
        trainable=True,
    ):
        """Add a new weight into the layer."""
        name = name if name else 'weights'
        shape = shape if shape is not None else []
        weight = init(shape=shape, trainable=trainable)
        weight._name = context.get_name_scope() + name
        if trainable is True:
            if self._trainable_weights is None:
                self._trainable_weights = []
            self._trainable_weights.append(weight)
        else:
            if self._nontrainable_weights is None:
                self._nontrainable_weights = []
            self._nontrainable_weights.append(weight)
        return weight

    def __call__(self, inputs, **kwargs):
        """The preprocessor for ``self.forward(...)``."""
        with context.name_scope(self.name):
            # Maybe build the layer at the first time.
            if not self._built:
                if isinstance(self, LayerList):
                    self._input_tensors = inputs
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


class LayerNode(object):
    """
    The class :class:`LayerNode` class represents a conceptional node for a layer.

    LayerNode is used for building static model and it is actually a light weighted
    wrapper over Layer. Specifically, it is used for building static computational graph
    (see _construct_graph() in tl.models.Model). In static model, each layer relates to
    one or more LayerNode, and the connection relationship between layers is built upon
    LayerNode. In addition, LayerNode eases layer reuse and weights sharing.

    Parameters
    ----------
    layer : tl.layers.Layer
        A tl layer that wants to create a node.
    node_index : int
        Index of this node in layer._nodes.
    in_nodes ：a list of LayerNode
        Father nodes to this node.
    in_tensors : a list of tensors
        Input tensors to this node.
    out_tensors : a list of tensors
        Output tensors to this node.
    in_tensor_idxes : a list of int
        Indexes of each input tensor in its corresponding node's out_tensors.

    Methods
    ---------
    __init__()
        Initializing the LayerNode.
    __call__()
        (1) Forwarding through the layer. (2) Update its input/output tensors.
    """

    def __init__(self, layer, node_index, in_nodes, in_tensors, out_tensors,
                 in_tensor_idxes):
        """

        Parameters
        ----------
        layer
        node_index
        in_nodes
        in_tensors
        out_tensors
        in_tensor_idxes
        """
        self.layer = layer
        self.node_index = node_index
        self.in_nodes = in_nodes
        self.out_nodes = []
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors
        self.name = layer.name + "_node_{}".format(node_index)

        self.in_tensors_idxes = in_tensor_idxes

        self.visited = False

    def __call__(self, inputs, **kwargs):
        """(1) Forwarding through the layer. (2) Update its input/output tensors."""
        outputs = self.layer.forward(inputs, **kwargs)
        self.in_tensors = nest.flatten(inputs)
        self.out_tensors = nest.flatten(outputs)
        return self.out_tensors


class LayerList(Layer):
    """Stack a group of layers into a sequential layer."""

    def __init__(self, layers, name=None):
        """Create a ``LayerList``.

        Parameters
        ----------
        layers : Sequence[dragon.vm.tensorlayer.layers.Layer]
            The layers to stack.
        name : str, optional
            The optional layer name.

        """
        super(LayerList, self).__init__(name=name)
        self._built = True
        self._all_layers = layers
        for layer in layers:
            if layer._built is False:
                self._built = False
            if layer._built and layer.all_weights is not None:
                if self._all_weights is None:
                    self._all_weights = []
                self._all_weights.extend(layer.all_weights)

    def build(self, input_shapes):
        """Build the layers sequentially."""
        inputs = self._input_tensors
        for layer in self._all_layers:
            built = layer._built
            outputs = layer.__call__(inputs)
            if not built and layer.all_weights is not None:
                if self._all_weights is None:
                    self._all_weights = []
                self._all_weights.extend(layer.all_weights)
            inputs = outputs

    def forward(self, inputs, *args, **kwargs):
        """Forward the computation sequentially."""
        outputs = inputs
        for layer in self._all_layers:
            outputs = layer.forward(outputs, *args, **kwargs)
        return outputs

    @Layer.training.setter
    def training(self, mode):
        """Set training mode."""
        self.training = mode
        for layer in self._all_layers:
            layer.training = mode

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return LayerList(list(self._all_layers)[idx])
        else:
            return self._all_layers[idx]

    def __len__(self):
        return len(self._all_layers)

    def __repr__(self):
        tmpstr = 'LayerList' + '(\n'
        for idx, layer in enumerate(self._all_layers):
            modstr = layer.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + str(idx) + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s
