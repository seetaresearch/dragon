# --------------------------------------------------------
# Caffe @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from collections import OrderedDict
from google.protobuf.text_format import Parse

import dragon.core.workspace as ws
from dragon.core.tensor import Tensor
import dragon.vm.theano as theano
import dragon.vm.theano.tensor as T

from .proto import caffe_pb2 as pb
from . import layers

class Blob(object):
    def __init__(self, tuple):
        self.data = tuple[0]; self.diff = tuple[1]


class Net(object):
    """
    Net supports the most exporting interfaces of ``caffe.Net``.

    We implement it completely in the python environment, which provides conveniences,

    especially when extending the modern architectures of `ConvNets`.
    """
    def __init__(self, *args):
        """Construct a Net by the ``prototxt`` file.

        Parameters
        ----------
        prototxt : str
            The path of ``.prototxt`` file.
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
        `NetInit(prototxt, phase)`_ - Construct a Net.

        `NetInitLoad(prototxt, model, phase)`_ - Construct a Net and load the model.

        """
        if len(args) == 2:
            self.NetInit(args[0], args[1])
        else: self.NetInitLoad(args[0], args[1], args[2])

    def NetInit(self, prototxt, phase='TRAIN'):
        """Construct a Net by the ``prototxt`` file.

        Parameters
        ----------
        prototxt : str
            The path of ``.prototxt`` file.
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
        Parse(open(prototxt,'r').read(), self._net)
        self._phase = phase
        self._layers = []
        if not hasattr(self, '_blobs'): self._blobs = {}
        self._params = {};
        self._swap_tensors = {}
        self._inputs_to_tensors = {}
        self._costs = []; self._wrts = []
        self._lr_mults = []; self._decay_mults = []

        if len(self._net.input) > 0:
            for input in self._net.input:
                if not input in self._blobs:
                    self._blobs[input] = {'data':Tensor(input).Variable(),
                                          'diff': Tensor(input + '_grad')}
                self._inputs_to_tensors[input] =  self._blobs[input]['data']

        for layer in self._net.layer:
            if not self.FilterLayer(layer): continue
            self._layers.append(getattr(layers, layer.type + 'Layer')(layer))

        self.Setup()

        for layer in self._net.layer:
            if not self.FilterLayer(layer): continue
            self.CheckBackward(layer)

    def NetInitLoad(self, prototxt, model, phase='TRAIN'):
        """Construct a Net by the ``prototxt`` file.

        Parameters
        ----------
        prototxt : str
            The path of ``.prototxt`` file.
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
        self.NetInit(prototxt, phase)
        self._model = model  # lazy-loading model

    def FilterLayer(self, LayerParameter):
        """Filter the layers.

        Parameters
        ----------
        LayerParameter : caffe_pb2.LayerParameter
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
            if not isinstance(outputs, list): outputs = [outputs]

            for idx, top in enumerate(layer._top):
                self._blobs[top] = {'data':outputs[idx], 'diff': Tensor(outputs[idx].name + '_grad')}
                self._net_outputs.add(top)

    def CheckBackward(self, LayerParameter):
        """Generate losses and learnable blobs.

        Parameters
        ----------
        LayerParameter : caffe_pb2.LayerParameter
            The parameter of ``Layer``.

        Returns
        -------
        None

        References
        ----------
        The implementation of `Init(net.cpp, L44)`_.

        """
        # append loss
        if LayerParameter.type.find('Loss') != -1:
            if len(LayerParameter.loss_weight) == 0:
                LayerParameter.loss_weight.extend([1.0])
            for idx, loss_weight in enumerate(LayerParameter.loss_weight):
                if loss_weight <= 0: continue
                self._costs.append(self.blobs[LayerParameter.top[idx]].data)
        else:
            if len(LayerParameter.loss_weight) != 0:
                for idx, loss_weight in enumerate(LayerParameter.loss_weight):
                    if loss_weight <= 0: continue
                    self._costs.append(self.blobs[LayerParameter.top[idx]].data)

        if self._phase != 'TRAIN': return

        # append param
        if len(LayerParameter.param) > 0:
            for idx, param in enumerate(LayerParameter.param):
                self._lr_mults.append(param.lr_mult if param.HasField('lr_mult') else 1.0)
                self._decay_mults.append(param.decay_mult if param.HasField('decay_mult') else 1.0)
                if self._lr_mults[-1] > 0:
                     self._wrts.append(self.params[LayerParameter.name][idx].data)

        # default ParamSpec
        elif len(self.params[LayerParameter.name]) > 0:
            for blob in self.params[LayerParameter.name]:
                self._wrts.append(blob.data)
                self._lr_mults.append(1.0)
                self._decay_mults.append(1.0)

    def function(self, givens=None):
        """Returns the function the ``ForwardBackward``.

        Parameters
        ----------
        givens : None or dict
            The givens to replace existing blobs.

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

        for cost in self._costs:
            for wrt in self._wrts:
                T.grad(cost, wrt)

        if givens is not None:
            if not isinstance(givens, dict):
                raise TypeError('The givens should be a dict.')
            for k, v in givens.items():
                if not isinstance(v, Tensor):
                    raise ValueError('The value of givens should be a Tensor.')
                self._swap_tensors[k] = v

        self._function = \
            theano.function(outputs=[self._blobs[name]['data']
                        for name in self._net_outputs], givens=self._swap_tensors)

        if hasattr(self, '_model'): ws.Restore(self._model, format='caffe')
        return self._function

    def share_with(self, other_net):
        """Share the parameters from another net. [**PyCaffe Style**]

        Parameters
        ----------
        other_net : Net
            The net to share with.

        Returns
        -------
        None

        References
        ----------
        The implementation of `ShareTrainedLayersWith(net.cpp, L665)`_.

        """
        if type(other_net) != type(self):
            raise TypeError('only type of Net can be shared.')

        other_params = other_net.params
        for name, blobs in self.params.items():
             if name in other_params:
                for idx, blob in enumerate(blobs):
                    self._swap_tensors[blob.data] = other_params[name][idx].data

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
        ws.Restore(model, format='caffe')

    def forward(self, **kwargs):
        """Forward pass. [**PyCaffe Style**]

        Parameters
        ----------
        inputs : dict or None
            The blobs to feed before.

        Returns
        -------
        Tensor or list of Tensor
            The outputs of the net.

        References
        ----------
        The implementation of `Net_forward(pycaffe.py, L88)`_.

        """
        def GetOutputs(net, net_outputs):
            ret = {}
            for output in net_outputs:
                ret[output] = ws.FetchTensor(net.blobs[output].data)
            return ret
        if kwargs:
            for name, blob in kwargs.items():
                ws.FeedTensor(self._inputs_to_tensors[name], blob)

        self.function()(return_outputs=False, stage='forward')
        return lambda net = self, net_outputs = self.outputs \
            : GetOutputs(net, net_outputs)

    def forward_v2(self, **kwargs):
        """Forward pass while silencing all net outputs.

        Parameters
        ----------
        inputs : dict or None
            The blobs to feed before.

        Returns
        -------
        None

        """
        if kwargs:
            for name, blob in kwargs.items():
                ws.FeedTensor(self._inputs_to_tensors[name], blob)
        self.function()(return_outputs=False, stage='forward')
        return None

    def backward(self, **kwargs):
        """Backward pass. [**PyCaffe Style**]

        Parameters
        ----------
        diffs : dict or None
            The diffs to feed before.

        Returns
        =------
        None

        References
        ----------
        The implementation of `Net_backward(pycaffe.py, L137)`_.

        """
        if kwargs:
            for name, blob in kwargs.items():
                ws.FeedTensor(self._blobs[name]['diff'].data, blob)

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
        tensors = []
        for layer in self._net.layer:
            if layer.name in self.params:
                for param in self.params[layer.name]:
                    tensors.append(param.data)
        ws.Snapshot(tensors, filename, suffix='', format='caffe')

    @property
    def blobs(self):
        """Return the blobs. [**PyCaffe Style**]

        Returns
        -------
        dict
            The format of dict is: <blob_name, blob>.

        Examples
        --------
        >>> data = net.blobs['conv1'].data
        >>> diff = net.blobs['conv1'].diff

        References
        ----------
        The implementation of `Net_blobs(pycaffe.py, L25)`_.

        """
        return OrderedDict([(name, Blob((blob['data'], blob['diff'])))
                                    for name, blob in self._blobs.items()])

    @property
    def params(self):
        """Return the parameters. [**PyCaffe Style**]

        Returns
        -------
        dict
            The format of dict is: <layer_name, list of blobs>.

        Examples
        --------
        >>> weights = net.params['conv1'][0].data
        >>> bias = net.params['conv1'][1].data

        References
        ----------
        The implementation of `Net_params(pycaffe.py, L58)`_.

        """
        return OrderedDict([(layer._name, [Blob((blob['data'],blob['diff']))
                                    for blob in layer._blobs]) for layer in self._layers])

    @property
    def lr_params(self):
        """Return the learnable parameters. [**PyCaffe Style**]

        Returns
        -------
        dict
            The format of dict is: <layer_name, list of blobs>.

        Examples
        --------
        >>> mean = net.params['bn1'][0].data
        >>> scale = net.lr_params['bn1'][0].data

        References
        ----------
        The extended implementation of `Net_params(pycaffe.py, L58)`_.

        """
        params = []
        for layer in self._layers:
            for blob in layer._blobs:
                if blob['diff'] is not None:
                    params.append(blob['data'])
        return params

    @property
    def store_params(self):
        """Return a list of parameters for snapshot. [**PyCaffe Style**]

        Returns
        -------
        list of Tensor
            The list of all parameters.

        Examples
        --------
        >>> import dragon.core.workspace as ws
        >>> ws.Snapshot(net.store_params(), filename='xxx', suffix='.caffeomdel')

        """
        params = []
        for layer in self._layers:
            for blob in layer._blobs:
                params.append(blob['data'])
        return params

    @property
    def inputs(self):
        """Return the inputs of net. [**PyCaffe Style**]

        Returns
        -------
        list of str
            The inputs.

        References
        ----------
        The implementation of `Net_inputs(pycaffe.py, L73)`_.

        """
        return [input for input in self._net.input]

    @property
    def outputs(self):
        """Return the outputs of net. [**PyCaffe Style**]

        Returns
        -------
        list of str
            The outputs

        References
        ----------
        The implementation of `Net_outputs(pycaffe.py, L81)`_.

        """
        return list(self._net_outputs)


    def replace(self, A, B):
        """Replace the A as B.

        Parameters
        ----------
        A : Tensor
            The A.
        B : Tensor
            The B.

        Returns
        -------
        None

        Examples
        --------
        >>> import dragon.ops as ops
        >>> data, label = ops.LMDBData()
        >>> net.replace(net.blobs['data'].data, data)
        >>> net.replace(net.blobs['label'].data, label)

        """
        self._swap_tensors[A] = B


class PartialNet(Net):
    """Construct a Net by explicitly injecting tensors.

    Examples
    --------
    >>> from dragon.core.tensor import Tensor
    >>> net = PartialNet('xxx.prototxt', 'TEST', **{'blob_name': Tensor().Variable()})

    """
    def __init__(self, *args, **kwargs):
        self._blobs = {}
        for input, tensor in kwargs.items():
            self._blobs[input] = {'data': tensor, 'diff': None}
        super(PartialNet, self).__init__(*args)
