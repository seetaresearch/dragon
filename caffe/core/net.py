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
"""The base net class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import google.protobuf.text_format
import numpy

from dragon.core.autograph import backprop
from dragon.core.autograph import context as eager_context
from dragon.core.autograph import function_impl
from dragon.core.framework import context
from dragon.core.util import nest
from dragon.core.util import serialization
from dragon.vm.caffe.core import layers as layer_factory
from dragon.vm.caffe.core.proto import caffe_pb2


class Blob(object):
    """Blob class."""

    def __init__(self, data, diff):
        self.data = data
        self.diff = diff

    @property
    def shape(self):
        """Return the blob shape."""
        return self.data.shape

    @property
    def count(self):
        """Return the blob size."""
        return self.data.size


class Net(object):
    """Base network class to connect layers."""

    def __init__(self, network_file, phase='TEST', weights=None):
        """Create a ``Net``.

        Parameters
        ----------
        network_file : str
            The path of text proto file to load network.
        phase : str, optional, default='TEST'
            The execution phase.
        weights : str, optional
            The path of binary proto file to load weights.

        """
        # Parse the network file.
        with open(network_file, 'r') as f:
            self._proto = google.protobuf.text_format.Parse(
                f.read(), caffe_pb2.NetParameter())
        self._phase = phase
        self._layers = []
        self._learnable_blobs = []
        self._net_blobs = dict()
        self._net_outputs = set()
        # Construct the layers from proto.
        layer_names = []
        for layer_param in self._proto.layer:
            if not self._filter_layer(layer_param):
                continue
            try:
                layer_index = layer_names.index(layer_param.name)
                call_layer = self._layers[layer_index]
            except ValueError:
                call_layer = None
                layer_names.append(layer_param.name)
            cls = getattr(layer_factory, layer_param.type)
            self._layers.append(cls(layer_param))
            self._layers[-1]._call_layer = call_layer
        # Add an input layer for the legacy inputs.
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
        # Connect layers to get outputs.
        self._init()
        # Load the pre-trained weights if necessary
        if weights is not None:
            self.copy_from(weights)

    @property
    def blobs(self):
        """Return the blob dict.

        Returns
        -------
        dict
            The blob dict.

        """
        if not hasattr(self, '_blob_dict'):
            self._blob_dict = collections.OrderedDict([
                (name, Blob(blob['data'], blob['diff']))
                for name, blob in self._net_blobs.items()])
        return self._blob_dict

    @property
    def params(self):
        """Return the parameter dict.

        Returns
        -------
        dict
            The parameter dict.

        """
        if not hasattr(self, '_param_dict'):
            self._param_dict = collections.OrderedDict()
            for layer in self._layers:
                if layer.name not in self._param_dict:
                    blobs = []
                    for blob in layer.blobs:
                        blobs.append(Blob(blob['data'], blob['diff']))
                        if 'decay_mult' in blob:
                            blobs[-1].decay_mult = blob['decay_mult']
                    self._param_dict[layer.name] = blobs
        return self._param_dict

    @property
    def inputs(self):
        """Return the input blob names.

        Returns
        -------
        Sequence[str]
            The input names.

        """
        if not hasattr(self, '_input_list'):
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
        if not hasattr(self, '_output_list'):
            self._output_list = list(self._net_outputs)
        return self._output_list

    def copy_from(self, other):
        """Copy layers from the other.

        Parameters
        ----------
        other : Union[str, NetParameter]
            The path of binary proto file or ``NetParameter``.

        """
        if (hasattr(other, 'ParseFromString') and
                callable(other.ParseFromString)):
            self.from_proto(other)
        else:
            self.from_proto(serialization.deserialize_proto(
                serialization.load_bytes(other), caffe_pb2.NetParameter()))

    def forward(self, **kwargs):
        """The forward pass."""
        for key, value in kwargs.items():
            if key in self._net_blobs:
                impl = self._net_blobs[key]['data']._impl
                impl.FromNumpy(numpy.array(value), False)
        self._compute_outputs()

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
        proto_bytes = serialization.serialize_proto(self.to_proto())
        serialization.save_bytes(proto_bytes, filepath)

    def to_proto(self):
        """Serialize to the proto.

        Returns
        -------
        NetParameter
            The ``NetParameter`` protocol buffer.

        """
        layer_proto = [layer.to_proto() for layer in self._layers]
        return caffe_pb2.NetParameter(name=self._proto.name, layer=layer_proto)

    def _filter_layer(self, layer_param):
        """Check if layer should be included."""
        if not layer_param.name:
            raise ValueError('Excepted non-empty layer name.')
        phase_dict = {'TRAIN': 0, 'TEST': 1}
        if (layer_param.HasField('phase') and
                layer_param.phase != phase_dict[self._phase]):
            return False
        for include in layer_param.include:
            if (include.HasField('phase') and
                    include.phase != phase_dict[self._phase]):
                return False
        layer_param.phase = phase_dict[self._phase]
        return True

    @function_impl.function
    def _compute_outputs(self, **kwargs):
        """Compute network outputs."""
        return [self.blobs[key].data for key in self.outputs]

    def _init(self):
        """Connect the layers sequentially."""
        losses, learnable_blobs = [], []
        grad_tape = backprop.GradientTape()
        # Collect bottom and top blobs.
        for i, layer in enumerate(self._layers):
            bottoms = []
            for bottom_name in layer.bottom:
                if bottom_name not in self._net_blobs:
                    raise RuntimeError('Bottom "{}" is unknown.'.format(bottom_name))
                bottoms.append(self._net_blobs[bottom_name])
                if bottom_name in self._net_outputs:
                    self._net_outputs.remove(bottom_name)
            if isinstance(layer, layer_factory.BatchNorm):
                next_layer = self._layers[i + 1]
                if isinstance(next_layer, layer_factory.Scale):
                    layer.scale_layer = next_layer
            with context.name_scope(layer.name), grad_tape:
                outputs = layer.setup([blob['data'] for blob in bottoms])
            if outputs is not None:
                outputs = nest.flatten(outputs)
                for j, top_name in enumerate(layer.top):
                    self._net_blobs[top_name] = {'data': outputs[j], 'diff': None}
                    self._net_outputs.add(top_name)
                loss_weights = list(layer._proto.loss_weight)
                if layer._proto.type.find('Loss') != -1:
                    if len(loss_weights) == 0:
                        loss_weights.append(1)
                for j, loss_weight in enumerate(loss_weights):
                    if loss_weight > 0:
                        losses.append(outputs[j])
            for j, blob in enumerate(layer.blobs):
                lr_mult, decay_mult = 1, 1
                if j < len(layer._proto.param):
                    p = layer._proto.param[j]
                    lr_mult = p.lr_mult if p.HasField('lr_mult') else 1
                    decay_mult = p.decay_mult if p.HasField('decay_mult') else 1
                if lr_mult > 0 and blob['data'].requires_grad:
                    if decay_mult == 0:
                        blob['data']._weight_decay = 0
                    learnable_blobs.append(blob)
        if self._phase == 'TRAIN':
            with eager_context.graph_mode():
                grads = grad_tape.gradient(
                    losses, [blob['data'] for blob in learnable_blobs])
            for blob, grad in zip(learnable_blobs, grads):
                blob['diff'] = grad
        # Collect all learnable blobs.
        for blobs in self.params.values():
            for blob in blobs:
                if blob.diff:
                    self._learnable_blobs.append(blob)
