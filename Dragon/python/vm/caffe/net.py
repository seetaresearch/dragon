# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from collections import OrderedDict
from google.protobuf.text_format import Parse
import dragon.core.workspace as ws
from dragon.core.tensor import Tensor
import dragon.vm.theano as theano
import dragon.vm.theano.tensor as T
import proto.caffe_pb2 as pb
import layers as Layer

class Blob(object):
    def __init__(self, tuple):
        self.data = tuple[0]; self.diff = tuple[1]

class Net(object):
    def __init__(self, *args):
        if len(args) == 2:
            self.NetInit(args[0], args[1])
        else: self.NetInitLoad(args[0], args[1], args[2])

    def NetInit(self, prototxt, phase='TRAIN'):
        self._net = pb.NetParameter()
        Parse(open(prototxt,'r').read(), self._net)
        self._phase = phase
        self._layers = []
        if not hasattr(self, '_blobs'): self._blobs = {}
        self._params = {}; self._swap_blobs = {}
        self._inputs_to_tensors = {}
        self._costs = []; self._wrts = []
        self._lr_mults = []; self._decay_mults = []

        if len(self._net.input) > 0:
            for input in self._net.input:
                if not self._blobs.has_key(input):
                    # create new tensors
                    self._blobs[input] = {'data':Tensor(input).Variable(),
                                          'diff': Tensor(input + '_grad')}
                self._inputs_to_tensors[input] =  self._blobs[input]['data']

        for layer in self._net.layer:
            if not self.FilterNet(layer): continue
            self._layers.append(getattr(Layer, layer.type + 'Layer')(layer))

        self.Setup()

        for layer in self._net.layer:
            if not self.FilterNet(layer): continue
            self.CheckBackward(layer)

    def NetInitLoad(self, prototxt, model, phase='TRAIN'):
        self.NetInit(prototxt, phase)
        self._model = model  # lazy-loading model

    def FilterNet(self, LayerParameter):
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
        # append loss
        if LayerParameter.type.find('Loss') != -1:
            if len(LayerParameter.loss_weight) == 0:
                LayerParameter.loss_weight.extend([1.0])
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

    @property
    def function(self):
        """ the CC Graph will create only once get this attr """
        if hasattr(self, '_function'): return self._function

        for cost in self._costs:
            for wrt in self._wrts:
                T.grad(cost, wrt)

        self._function = \
            theano.function(outputs=[self._blobs[name]['data']
                        for name in self._net_outputs], swaps=self._swap_blobs)

        if hasattr(self, '_model'): ws.Restore(self._model, format=1)
        return self._function

    def share_with(self, other_net):
        """ simply follow the pycaffe style """
        if type(other_net) != type(self):
            raise TypeError('only type of Net can be shared.')

        other_params = other_net.params
        for name, blobs in self.params.iteritems():
             if other_params.has_key(name):
                for idx, blob in enumerate(blobs):
                    self._swap_blobs[blob.data] = other_params[name][idx].data

    def copy_from(self, model):
        """ simply follow the pycaffe style """
        ws.Restore(model, format=1)

    def forward(self, **kwargs):
        """ simply follow the pycaffe style """
        def GetOutputs(net, net_outputs):
            ret = {}
            for output in net_outputs:
                ret[output] = ws.FetchTensor(net.blobs[output].data)
            return ret
        if kwargs:
            for name, blob in kwargs.iteritems():
                ws.FeedTensor(self._inputs_to_tensors[name], blob)

        self.function(return_outputs=False, stage='forward')
        return lambda net = self, net_outputs = self.outputs \
            : GetOutputs(net, net_outputs)

    def backward(self):
        """ simply follow the pycaffe style """
        self.function(return_outputs=False, stage='backward')

    def save(self, filename, suffix='.caffemodel'):
        """ simply follow the pycaffe style """
        if not hasattr(self, '_function'): func = self.function
        tensors = []
        for layer in self._net.layer:
            if self.params.has_key(layer.name):
                for param in self.params[layer.name]:
                    tensors.append(param.data)

        ws.Snapshot(tensors, filename, suffix=suffix, format=1)

    @property
    def layers(self):
        """ simply follow the pycaffe style """
        return OrderedDict([(layer._name, layer) for layer in self._layers])

    @property
    def blobs(self):
        """ simply follow the pycaffe style """
        return OrderedDict([(name,Blob((blob['data'], blob['diff'])))
                                    for name, blob in self._blobs.iteritems()])
    @property
    def params(self):
        """ simply follow the pycaffe style """
        return OrderedDict([(layer._name, [Blob((blob['data'],blob['diff']))
                                    for blob in layer._blobs]) for layer in self._layers])
    @property
    def lr_params(self):
        params = []
        for layer in self._layers:
            for blob in layer._blobs:
                if blob['diff'] is not None:
                    params.append(blob['data'])
        return params

    @property
    def store_params(self):
        params = []
        for layer in self._layers:
            for blob in layer._blobs:
                params.append(blob['data'])
        return params

    @property
    def inputs(self):
        """ simply follow the pycaffe style """
        return [input for input in self._net.input]

    @property
    def outputs(self):
        """ simply follow the pycaffe style """
        return self._net_outputs

    def replace(self, old, new):
        self._swap_blobs[old] = new

class PartialNet(Net):
    def __init__(self, *args, **kwargs):
        self._blobs = {}
        for input, tensor in kwargs.iteritems():
            self._blobs[input] = {'data': tensor, 'diff': None}
        super(PartialNet, self).__init__(*args)
