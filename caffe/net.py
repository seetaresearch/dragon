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

"""Implementation for the ``Net`` C++ class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from google.protobuf import text_format

from dragon.core.autograph import def_function
from dragon.core.autograph import grad_impl
from dragon.core.autograph.tensor import RefTensor
from dragon.core.autograph.tensor import Tensor
from dragon.core.framework import workspace
from dragon.core.util import nest
from dragon.vm.caffe import layers as layer_factory
from dragon.vm.caffe.proto import caffe_pb2


class Blob(object):
    def __init__(self, tuple):
        self.data, self.diff = tuple[0], tuple[1]
        self.lr_multiplier = self.decay_multiplier = 1.


class Net(object):
    """The abstraction ``caffe.Net``.

    This class accepts a proto-text file, and an optional
    serialized model weights. You can also specify a phase
    flag to indicate whether to compute the gradients:

    ```python
    train_net = Net('train.prototxt', 'TRAIN')
    test_net = Net('test.prototxt', 'my.caffemodel', 'TEST')
    ```

    """

    def __init__(self, *args):
        """Create a Net.

        Parameters
        ----------
        network_file : str
            The path of ``net.prototxt`` file.
        weights : str, optional
            The path of the weights file.
        phase : {'TRAIN', 'TEST'}, optional
            The optional phase.

        """
        if len(args) == 2:
            (net_file, self._phase), weights = args, None
        elif len(args) == 3:
            net_file, weights, self._phase = args
        else:
            raise ValueError('Excepted 2 or 3 args.')
        self._net_proto = caffe_pb2.NetParameter()
        self._blobs = {}
        self._layers = []
        self._layer_blobs = []
        self._losses = []
        self._variables = []

        self._blob_dict = None
        self._param_dict = None
        self._input_list = None
        self._output_list = None

        with open(net_file, 'r') as f:
            text_format.Parse(f.read(), self._net_proto)

        if len(self._net_proto.input) > 0:
            shapes = self._net_proto.input_shape
            for i, input in enumerate(self._net_proto.input):
                shape = [e for e in shapes[i].dim] if i < len(shapes) else None
                if input not in self._blobs:
                    data = Tensor(input, shape=shape, dtype='float32').placeholder()
                    self._blobs[input] = {
                        'data': data,
                        'diff': RefTensor(
                            data.id + '_grad',
                            shape=shape,
                            dtype=data.dtype
                        ),
                    }

        for layer in self._net_proto.layer:
            if not self._filter_layer(layer):
                continue
            cls = getattr(layer_factory, layer.type)
            self._layers.append(cls(layer))

        self._setup()

        for layer in self._net_proto.layer:
            if not self._filter_layer(layer):
                continue
            self._collect_losses_and_variables(layer)

        if weights is not None:
            workspace.load(weights, format='caffe')

    def _filter_layer(self, layer_param):
        """Indicate whether the given layer should be included."""
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

    def _setup(self):
        """Connect the layers sequentially."""
        self._net_outputs = set()
        # Collect bottom and top blobs.
        for layer in self._layers:
            bottom = []
            for blob in layer._bottom:
                if blob not in self._blobs:
                    raise RuntimeError('bottom({}) is unknown.'.format(blob))
                bottom.append(self._blobs[blob])
                if blob in self._net_outputs:
                    self._net_outputs.remove(blob)

            outputs = layer.setup([blob['data'] for blob in bottom])
            outputs = nest.flatten(outputs)

            for i, blob in enumerate(layer._top):
                self._blobs[blob] = {
                    'data': outputs[i],
                    'diff': RefTensor(outputs[i].id + '_grad'),
                }
                self._net_outputs.add(blob)

        # Collect layer param blobs.
        for blobs in self.params.values():
            self._layer_blobs.extend(blobs)

    def _collect_losses_and_variables(self, layer_param):
        """Collect losses and variables."""
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
                    self._variables.append(blob.data)
        else:
            for blob in self.params[layer_param.name]:
                if blob.diff is not None and blob.lr_multiplier > 0:
                    self._variables.append(blob.data)

    @classmethod
    def copy_from(cls, weights):
        """Copy the weights from the binary proto file.

        Parameters
        ----------
        weights : str
            The path of the weights file.

        """
        workspace.load(weights, format='caffe')

    @def_function.function
    def forward_backward(self, **kwargs):
        """Forward pass following by backward pass.

        This function will be compiled to a computation graph
        once executed, with implicit feeding of inputs.

        """
        grad_impl.gradients(self._losses, self._variables)
        return [self.blobs[key].data for key in self.outputs]

    def forward(self, **inputs):
        """Forward pass.

        Parameters
        ----------
        inputs : dict, optional
            The blobs to feed.

        Returns
        -------
        callable
            The callable to return outputs.

        """
        for name, blob in inputs.items():
            workspace.feed_tensor(self._blobs[name]['data'], blob)
        self.forward_backward(return_outputs=False, stage='forward')
        return lambda: dict(
            (output, self.blobs[output].data.get_value())
            for output in self.outputs
        )

    def backward(self, **diffs):
        """Backward pass.

        Parameters
        ----------
        diffs : dict, optional
            The diffs to feed.

        """
        for name, blob in diffs.items():
            workspace.feed_tensor(self.blobs[name].diff, blob)
        self.forward_backward(return_outputs=False, stage='backward')

    def save(self, filename):
        """Save the parameters into a binary file.

        Parameters
        ----------
        filename : str
            The path of model file.

        """
        workspace.save(
            tensors=[blob.data for blob in self._layer_blobs],
            filename=filename, suffix='', format='caffe',
        )

    @property
    def blobs(self):
        """Return the blob dict.

        Blobs stored in the dict will be:

        ```python
        for blob_name, blob in net.blobs():
            print(blob.data)  # DataTensor
            print(blob.diff)  # GradTensor
        ```

        Returns
        -------
        Dict
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

        Parameters stored in the dict will be:

        ```python
        for layer_name, blobs in net.params():
            print(layer_name)
            for blob in blobs:
                print('  *', blob.data)  # DataTensor
                print('  *', blob.diff)  # GradTensor
        ```

        Returns
        -------
        Dict
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
            self._input_list = [input for input in self._net_proto.input]
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
