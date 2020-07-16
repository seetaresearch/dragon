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
"""The base net class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from google.protobuf import text_format

from dragon.core.autograph import def_function
from dragon.core.autograph import grad_impl
from dragon.core.autograph.tensor import TensorRef
from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.util import nest
from dragon.core.util import serialization
from dragon.vm.caffe import layers as layer_factory
from dragon.vm.caffe.proto import caffe_pb2


class Blob(object):
    def __init__(self, tuple):
        self.data, self.diff = tuple[0], tuple[1]
        self.lr_multiplier = self.decay_multiplier = 1.


class Net(object):
    """The base net class to connect layers.

    This class accepts a network file, and an optional parameter file.
    Besides, a phase tag is required to compute gradients or not:

    ```python
    net1 = caffe.Net('train.prototxt', 'TRAIN')
    net2 = caffe.Net('test.prototxt', 'test.caffemodel', 'TEST')
    ```

    """

    def __init__(self, *args):
        """Create a ``Net``.

        Parameters
        ----------
        net_file : str
            The path of text proto file to load network.
        param_file : str, optional
            The path of binary proto file to load parameters.
        phase : {'TRAIN', 'TEST'}, optional
            The optional phase tag.

        """
        if len(args) == 2:
            (net_file, self._phase), param_file = args, None
        elif len(args) == 3:
            net_file, param_file, self._phase = args
        else:
            raise ValueError('Excepted 2 or 3 args.')
        self._blobs = {}
        self._layers = []
        self._layer_blobs = []
        self._losses = []
        self._params = []
        self._blob_dict = None
        self._param_dict = None
        self._input_list = None
        self._output_list = None
        # Parse the network file
        with open(net_file, 'r') as f:
            self._proto = text_format.Parse(f.read(), caffe_pb2.NetParameter())
        # Construct the layer class from proto
        for layer_param in self._proto.layer:
            if not self._filter_layer(layer_param):
                continue
            cls = getattr(layer_factory, layer_param.type)
            with context.name_scope(layer_param.name):
                self._layers.append(cls(layer_param))
        # Prepare for the legacy net inputs
        if len(self._proto.input) > 0:
            layer_param = caffe_pb2.LayerParameter(
                name='data',
                type='Input',
                top=self._proto.input,
                input_param=caffe_pb2.InputParameter(
                    shape=self._proto.input_shape))
            cls = getattr(layer_factory, layer_param.type)
            with context.name_scope(layer_param.name):
                self._layers.insert(0, cls(layer_param))
        # Call layers sequentially to get outputs
        self._setup()
        # Collect losses and parameters
        for layer in self._proto.layer:
            if not self._filter_layer(layer):
                continue
            self._collect_losses_and_params(layer)
        # Load the pre-trained weights if necessary
        if param_file is not None:
            self.copy_from(param_file)

    @property
    def blobs(self):
        """Return the blob dict.

        Returns
        -------
        dict
            The blob dict.

        """
        if self._blob_dict is None:
            self._blob_dict = collections.OrderedDict([
                (name, Blob((blob['data'], blob['diff'])))
                for name, blob in self._blobs.items()])
        return self._blob_dict

    @property
    def params(self):
        """Return the parameter dict.

        Returns
        -------
        dict
            The parameter dict.

        """
        if self._param_dict is None:
            self._param_dict = collections.OrderedDict([
                (layer._name, [
                    Blob((blob['data'], blob['diff']))
                    for blob in layer._blobs]
                 ) for layer in self._layers])
        return self._param_dict

    @property
    def losses(self):
        """Return the losses.

        Returns
        -------
        Sequence[dragon.Tensor]
            The losses.

        """
        return self._losses

    @property
    def inputs(self):
        """Return the input blob names.

        Returns
        -------
        Sequence[str]
            The input names.

        """
        if self._input_list is None:
            self._input_list = [input for input in self._proto.input]
        return self._input_list

    @property
    def outputs(self):
        """Return the output blob names.

        Returns
        -------
        Sequence[str]
            The output names.

        """
        if self._output_list is None:
            self._output_list = list(self._net_outputs)
        return self._output_list

    def backward(self, **diffs):
        """The backward pass.

        Parameters
        ----------
        diffs : dict, optional
            The data to feed to the diffs.

        """
        current_ws = workspace.get_workspace()
        for name, blob in diffs.items():
            current_ws.feed_tensor(self.blobs[name].diff, blob)
        self._forward_backward_impl(executing_stage='backward')

    def copy_from(self, other):
        """Copy layers from the other.

        Parameters
        ----------
        other : Union[str, NetParameter]
            The path of binary proto file or ``NetParameter``.

        """
        if hasattr(other, 'ParseFromString') and \
                callable(other.ParseFromString):
            self.from_proto(other)
        else:
            self.from_proto(serialization.deserialize_proto(
                serialization.load_bytes(other), caffe_pb2.NetParameter()))

    def forward(self, **inputs):
        """The forward pass.

        Parameters
        ----------
        inputs : dict, optional
            The data to feed to the inputs.

        Returns
        -------
        callable
            The callable to fetch outputs.

        """
        current_ws = workspace.get_workspace()
        for name, blob in inputs.items():
            current_ws.feed_tensor(self._blobs[name]['data'], blob)
        self._forward_backward_impl(executing_stage='forward')
        return lambda: dict(
            (output, current_ws.fetch_tensor(self.blobs[output].data))
            for output in self.outputs)

    def forward_backward(self, **inputs):
        """The forward and backward pass.

        Parameters
        ----------
        inputs : dict, optional
            The data to feed to the inputs.

        Returns
        -------
        callable
            The callable to fetch outputs.

        """
        current_ws = workspace.get_workspace()
        for name, blob in inputs.items():
            current_ws.feed_tensor(self._blobs[name]['data'], blob)
        self._forward_backward_impl()
        return lambda: dict(
            (output, current_ws.fetch_tensor(self.blobs[output].data))
            for output in self.outputs)

    def from_proto(self, proto):
        """Deserialize from the proto.

        Parameters
        ----------
        proto : NetParameter
            The ``NetParameter`` protocol buffer.

        """
        layer_dict = dict((layer.name, layer) for layer in proto.layer)
        for layer in self._layers:
            if layer.name in layer_dict:
                layer.from_proto(layer_dict[layer.name])

    def save(self, filepath):
        """Save proto into a binary file.

        Parameters
        ----------
        filepath : str
            The path of binary proto file.

        """
        serialization.save_bytes(
            serialization.serialize_proto(
                self.to_proto()), filepath)

    def to_proto(self):
        """Serialize to the proto.

        Returns
        -------
        NetParameter
            The ``NetParameter`` protocol buffer.

        """
        return caffe_pb2.NetParameter(
            name=self._proto.name,
            layer=[layer.to_proto() for layer in self._layers])

    def _collect_losses_and_params(self, layer_param):
        """Collect losses and parameters."""
        if layer_param.type.find('Loss') != -1:
            if len(layer_param.loss_weight) == 0:
                layer_param.loss_weight.extend([1.])
            for i, loss_weight in enumerate(layer_param.loss_weight):
                if loss_weight <= 0:
                    continue
                self._losses.append(self.blobs[layer_param.top[i]].data)
        else:
            if len(layer_param.loss_weight) != 0:
                for i, loss_weight in enumerate(layer_param.loss_weight):
                    if loss_weight <= 0:
                        continue
                    self._losses.append(self.blobs[layer_param.top[i]].data)
        if self._phase != 'TRAIN':
            return
        if len(layer_param.param) > 0:
            for i, p in enumerate(layer_param.param):
                blob = self.params[layer_param.name][i]
                blob.lr_multiplier = p.lr_mult if p.HasField('lr_mult') else 1.
                blob.decay_multiplier = p.decay_mult if p.HasField('decay_mult') else 1.
                if blob.diff is not None and blob.lr_multiplier > 0:
                    self._params.append(blob.data)
        else:
            for blob in self.params[layer_param.name]:
                if blob.diff is not None and blob.lr_multiplier > 0:
                    self._params.append(blob.data)

    def _filter_layer(self, layer_param):
        """Check if layer should be included."""
        phase_dict = {'TRAIN': 0, 'TEST': 1}
        if layer_param.HasField('phase') and \
                layer_param.phase != phase_dict[self._phase]:
            return False
        for include in layer_param.include:
            if include.HasField('phase') and \
                    include.phase != phase_dict[self._phase]:
                return False
        layer_param.phase = phase_dict[self._phase]
        return True

    @def_function.function
    def _forward_backward_impl(self, **kwargs):
        """Implementation for ``self.forward_backward(...)``."""
        grad_impl.gradients(self._losses, self._params)
        return [self.blobs[key].data for key in self.outputs]

    def _setup(self):
        """Connect the layers sequentially."""
        self._net_outputs = set()
        # Collect bottom and top blobs.
        for layer_idx, layer in enumerate(self._layers):
            bottom = []
            for blob in layer._bottom:
                if blob not in self._blobs:
                    raise RuntimeError('bottom({}) is unknown.'.format(blob))
                bottom.append(self._blobs[blob])
                if blob in self._net_outputs:
                    self._net_outputs.remove(blob)
            if isinstance(layer, layer_factory.BatchNorm):
                next_layer = self._layers[layer_idx + 1]
                if isinstance(next_layer, layer_factory.Scale):
                    layer.fuse_with_scale_layer(next_layer)
            with context.name_scope(layer._name):
                outputs = layer.setup([blob['data'] for blob in bottom])
            if outputs is not None:
                outputs = nest.flatten(outputs)
                for blob_idx, blob in enumerate(layer._top):
                    self._blobs[blob] = {
                        'data': outputs[blob_idx],
                        'diff': TensorRef(outputs[blob_idx].id + '_grad')}
                    self._net_outputs.add(blob)
        # Collect layer param blobs.
        for blobs in self.params.values():
            self._layer_blobs.extend(blobs)
