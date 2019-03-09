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

"""The implementation of the ``Net`` C++ class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon

from collections import OrderedDict
from google.protobuf.text_format import Parse as parse_text_proto
from dragon.vm.caffe import layers as layer_factory
from dragon.vm.caffe.proto import caffe_pb2 as pb


class Blob(object):
    def __init__(self, tuple):
        self.data, self.diff = tuple[0], tuple[1]
        self.lr_multiplier = self.decay_multiplier = 1.0


class Net(object):
    """Net supports the most exporting interfaces of ``caffe.Net``.

    We implement it completely in the python environment, which provides conveniences,

    especially when extending the modern architectures of ``Convolution Networks``,

    """
    def __init__(self, *args):
        """Construct a Net by the ``proto_txt`` file.

        Parameters
        ----------
        proto_txt : str
            The path of ``.proto_txt`` file.
        model : str
            (Optional) The path of the ``.caffemodel`` file.
        phase : str
            The phase, ``TRAIN`` or ``TEST``.

        Returns
        -------
        Net
            The net.

        Examples
        --------
        >>> train_net = Net('train.prototxt', 'TRAIN')
        >>> test_net = Net('test.prototxt', 'snapshot/xxx.caffemodel', 'TEST')

        References
        ----------
        `NetInit(proto_txt, phase)`_ - Construct a Net.

        `NetInitLoad(proto_txt, model, phase)`_ - Construct a Net and load the model.

        """
        if len(args) == 2: self.NetInit(args[0], args[1])
        else: self.NetInitLoad(args[0], args[1], args[2])

    def NetInit(self, proto_txt, phase='TRAIN'):
        """Construct a Net by the ``proto_txt`` file.

        Parameters
        ----------
        proto_txt : str
            The path of ``proto_txt`` file.
        phase : str
            The phase, ``TRAIN`` or ``TEST``.

        Returns
        -------
        Net
            The net.

        References
        ----------
        The implementation of `Net_Init(_caffe.cpp, L109)`_.

        """
        self._net = pb.NetParameter()
        parse_text_proto(open(proto_txt,'r').read(), self._net)
        self._phase = phase
        self._layers = []
        self._inputs_to_tensors = {}
        if not hasattr(self, '_blobs'): self._blobs = {}
        self._losses, self._trainable_vars = [], []

        if len(self._net.input) > 0:
            for input in self._net.input:
                if not input in self._blobs:
                    variable = dragon.Tensor(input).Variable()
                    self._blobs[input] = {
                        'data': variable,
                        'diff': dragon.Tensor.Ref(variable.name + '_grad'),
                    }
                self._inputs_to_tensors[input] = self._blobs[input]['data']

        for layer in self._net.layer:
            if not self.FilterLayer(layer): continue
            self._layers.append(getattr(layer_factory, layer.type + 'Layer')(layer))

        self.Setup()

        for layer in self._net.layer:
            if not self.FilterLayer(layer): continue
            self.CheckBackward(layer)

    def NetInitLoad(self, proto_txt, model, phase='TRAIN'):
        """Construct a Net by the ``proto_txt`` file.

        Parameters
        ----------
        proto_txt : str
            The path of ``proto_txt`` file.
        model : str
            (Optional) The path of the ``.caffemodel`` file.
        phase : str
            The phase, ``TRAIN`` or ``TEST``.

        Returns
        -------
        Net
            The net.

        References
        ----------
        The implementation of `Net_Init_Load(_caffe.cpp, L137)`_.

        """
        self.NetInit(proto_txt, phase)
        self._model = model  # lazy-loading model

    def FilterLayer(self, LayerParameter):
        """Filter the layers.

        Parameters
        ----------
        LayerParameter : LayerParameter
            The parameter of ``Layer``.

        Returns
        -------
        boolean
            Whether this layer should be keep.

        References
        ----------
        The implementation of `FilterNet(net.cpp, L259)`_.

        """
        CAFFE_PHASE = {'TRAIN': 0, 'TEST': 1}
        if LayerParameter.HasField('phase') and \
                LayerParameter.phase != CAFFE_PHASE[self._phase]:
                    return False
        for include in LayerParameter.include:
            if include.HasField('phase') and \
                include.phase != CAFFE_PHASE[self._phase]:
                    return False
        LayerParameter.phase = CAFFE_PHASE[self._phase]
        return True

    def Setup(self):
        """Setup the net.

        Returns
        -------
        None

        References
        ----------
        The implementation of `Init(net.cpp, L44)`_.

        """
        self._net_outputs = set()
        for layer in self._layers:
            bottom = []
            for bottom_name in layer._bottom:
                if not bottom_name in self._blobs:
                    raise RuntimeError('bottom({}) is unknown.'.format(bottom_name))
                bottom.append(self._blobs[bottom_name])
                if bottom_name in self._net_outputs:
                    self._net_outputs.remove(bottom_name)

            outputs = layer.Setup([blob['data'] for blob in bottom])
            if not isinstance(outputs, (list, tuple)): outputs = [outputs]

            for idx, top in enumerate(layer._top):
                self._blobs[top] = {
                    'data': outputs[idx],
                    'diff': dragon.Tensor.Ref(outputs[idx].name + '_grad'),
                }
                self._net_outputs.add(top)

    def CheckBackward(self, layer_param):
        """Generate losses and trainable blobs.

        Parameters
        ----------
        layer_param : LayerParameter
            The parameter of ``Layer``.

        Returns
        -------
        None

        References
        ----------
        The implementation of `Init(net.cpp, L44)`_.

        """
        # Append loss
        if layer_param.type.find('Loss') != -1:
            if len(layer_param.loss_weight) == 0:
                layer_param.loss_weight.extend([1.0])
            for i, loss_weight in enumerate(layer_param.loss_weight):
                if loss_weight <= 0: continue
                self._losses.append(self.blobs[layer_param.top[i]].data)
        else:
            if len(layer_param.loss_weight) != 0:
                for i, loss_weight in enumerate(layer_param.loss_weight):
                    if loss_weight <= 0: continue
                    self._losses.append(self.blobs[layer_param.top[i]].data)

        if self._phase != 'TRAIN': return

        # Append param
        if len(layer_param.param) > 0:
            for i, p in enumerate(layer_param.param):
                blob = self.params[layer_param.name][i]
                blob.lr_multiplier = p.lr_mult if p.HasField('lr_mult') else 1.0
                blob.decay_multiplier = p.decay_mult if p.HasField('decay_mult') else 1.0
                if blob.diff is not None and blob.lr_multiplier > 0:
                     self._trainable_vars.append(blob.data)

        # Default ParamSpec
        else:
            for blob in self.params[layer_param.name]:
                if blob.diff is not None and blob.lr_multiplier > 0:
                    self._trainable_vars.append(blob.data)

    def function(self):
        """Returns the function the ``ForwardBackward``.

        Returns
        -------
        lambda
            The function.

        See Also
        --------
        `theano.function(*args, **kwargs)`_ - How to make a graph. [**Theano Style**]

        References
        ----------
        The implementation of `ForwardBackward(net.cpp, L85)`_.

        """
        if hasattr(self, '_function'): return self._function

        for loss in self.losses:
            for var in self.trainable_variables:
                dragon.grad(loss, var)

        self._function = dragon.function(
            outputs=[self.blobs[key].data
                    for key in self.outputs])

        if hasattr(self, '_model'):
            dragon.workspace.Restore(self._model, format='caffe')

        return self._function

    def copy_from(self, model):
        """Copy the parameters from the binary proto file. [**PyCaffe Style**]

        Parameters
        ----------
        model : str
            The path of the ``.caffemodel`` file.

        See Also
        --------
        `workspace.Restore(*args, **kwargs)`_ - How to restore tensors from a file.

        References
        ----------
        The implementation of `CopyTrainedLayersFromBinaryProto(net.cpp, L780)`_.

        """
        dragon.workspace.Restore(model, format='caffe')

    def forward(self, **kwargs):
        """Forward pass. [**PyCaffe Style**]

        Parameters
        ----------
        inputs : dict, optional
            The blobs to feed before.

        Returns
        -------
        sequence of Tensor
            The outputs of the net.

        References
        ----------
        The implementation of `Net_forward(pycaffe.py, L88)`_.

        """
        def GetOutputs(net, net_outputs):
            ret = {}
            for output in net_outputs:
                ret[output] = dragon.workspace.FetchTensor(net.blobs[output].data)
            return ret

        for name, blob in kwargs.items():
            dragon.workspace.FeedTensor(self._inputs_to_tensors[name], blob)

        self.function()(return_outputs=False, stage='forward')

        return lambda net = self, net_outputs = self.outputs \
            : GetOutputs(net, net_outputs)

    def forward_v2(self, **kwargs):
        """Forward pass while silencing all net outputs.

        Parameters
        ----------
        inputs : dict, optional
            The blobs to feed before.

        Returns
        -------
        None

        """
        for name, blob in kwargs.items():
            dragon.workspace.FeedTensor(self._inputs_to_tensors[name], blob)
        self.function()(return_outputs=False, stage='forward')

    def backward(self, **kwargs):
        """Backward pass. [**PyCaffe Style**]

        Parameters
        ----------
        diffs : dict, optional
            The diffs to feed before.

        Returns
        -------
        None

        References
        ----------
        The implementation of `Net_backward(pycaffe.py, L137)`_.

        """
        for name, blob in kwargs.items():
            dragon.workspace.FeedTensor(self.blobs[name].diff, blob)
        self.function()(return_outputs=False, stage='backward')

    def save(self, filename):
        """Save the parameters into a binary file. [**PyCaffe Style**]

        Parameters
        ----------
        filename : str
            The path of model file.

        Returns
        -------
        None

        See Also
        --------
        `workspace.Snapshot(*args, **kwargs)`_ - How to snapshot tensors into a file.

        References
        ----------
        The implementation of `Net_Save(_caffe.cpp, L153)`_.

        """
        keys = set(); tensors = []
        for layer in self._net.layer:
            if layer.name in self.params:
                for param in self.params[layer.name]:
                    if param.data.name not in keys:
                        tensors.append(param.data)
                        keys.add(param.data.name)
        dragon.workspace.Snapshot(tensors, filename, suffix='', format='caffe')

    @property
    def blobs(self):
        """Return the blobs. [**PyCaffe Style**]

        Returns
        -------
        dict
            The format of dict is: <blob_name, blob>.

        Examples
        --------
        >>> data = self..blobs['conv1'].data
        >>> diff = self.blobs['conv1'].diff

        References
        ----------
        The implementation of `Net_blobs(pycaffe.py, L25)`_.

        """
        if not hasattr(self, '_blobs_dict'):
            self._blobs_dict = OrderedDict([
                (name, Blob((blob['data'], blob['diff'])))
                    for name, blob in self._blobs.items()])
        return self._blobs_dict

    @property
    def params(self):
        """Return the parameters. [**PyCaffe Style**]

        Returns
        -------
        dict
            The format of dict is: <layer_name, list of blobs>.

        Examples
        --------
        >>> weights = self..params['conv1'][0].data
        >>> bias = self.params['conv1'][1].data

        References
        ----------
        The implementation of `Net_params(pycaffe.py, L58)`_.

        """
        if not hasattr(self, '_params_dict'):
            self._params_dict = OrderedDict([
                (layer._name, [Blob((blob['data'], blob['diff']))
                    for blob in layer._blobs]) for layer in self._layers])
        return self._params_dict

    @property
    def losses(self):
        """Return the losses of this net.

        Returns
        -------
        sequence of Tensor
            The trainable parameters.

        """
        return self._losses

    @property
    def trainable_variables(self):
        """Return the trainable variables of this net.

        Returns
        -------
        sequence of Tensor
            The trainable parameters.

        """
        return self._trainable_vars

    @property
    def variables(self):
        """Return a list of parameters for snapshot. [**PyCaffe Style**]

        Returns
        -------
        sequence of Tensor
            The list of all parameters.

        Examples
        --------
        >>> import dragon
        >>> dragon.workspace.Snapshot(self.variables, filename='xxx', suffix='.caffeomdel')

        """
        return [(blob for blob in layer._blobs) for layer in self._layers]

    @property
    def inputs(self):
        """Return the inputs of net. [**PyCaffe Style**]

        Returns
        -------
        sequence of str
            The inputs.

        References
        ----------
        The implementation of `Net_inputs(pycaffe.py, L73)`_.

        """
        if not hasattr(self, '_input_list'):
            self._input_list = [input for input in self._net.input]
        return self._input_list

    @property
    def outputs(self):
        """Return the outputs of net. [**PyCaffe Style**]

        Returns
        -------
        sequence of str
            The outputs

        References
        ----------
        The implementation of `Net_outputs(pycaffe.py, L81)`_.

        """
        if not hasattr(self, '_output_list'):
            self._output_list = list(self._net_outputs)
        return self._output_list


class PartialNet(Net):
    """Construct a Net by explicitly injecting tensors.

    Examples
    --------
    >>> import dragon
    >>> net = PartialNet('xxx.proto_txt', 'TEST', **{'blob_name': dragon.Tensor().Variable()})

    """
    def __init__(self, *args, **kwargs):
        self._blobs = {}
        for input, tensor in kwargs.items():
            self._blobs[input] = {'data': tensor, 'diff': None}
        super(PartialNet, self).__init__(*args)