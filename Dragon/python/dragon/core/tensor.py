# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.config as config
import dragon.core.workspace as ws
import dragon.protos.dragon_pb2 as pb
import numpy as np
from collections import OrderedDict
from dragon.core.utils import MakeOperatorDef
from dragon.core.scope import GetOperatorName, GetTensorName
from six.moves import range as xrange


class Tensor(object):
    REGISTERED_FILLERS = {'Constant', 'Normal', 'TruncatedNormal',
                          'Uniform',  'Gaussian', 'Xavier',
                          'Variable'}

    def __init__(self, name=None, shape=None):
        self.name = name
        self.shape = shape

# ------------------------  Properties  ------------------------

    @property
    def expressions(self):
        if not hasattr(self,'_expressions'): self._expressions = {}
        return self._expressions

    @expressions.setter
    def expressions(self, value):
        assert isinstance(value, dict)
        self._expressions = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        from .scope import TENSOR_SCOPE
        if value is None: self._name = TENSOR_SCOPE + GetTensorName()
        else: self._name = TENSOR_SCOPE + value

    @property
    def grad_wrts(self):
        if not hasattr(self,'_grad_wrts'): self._grad_wrts = []
        return self._grad_wrts

    @property
    def grad_objs(self):
        if not hasattr(self, '_grad_objs'): self._grad_objs = []
        return self._grad_objs

    @property
    def shape(self):
        if not hasattr(self, '_shape'): self._shape = None
        return self._shape

    @shape.setter
    def shape(self, value):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError('tensor shape must be a list.')
        self._shape = value

    @property
    def extra_targets(self):
        if not hasattr(self,'_extra_targets'): self._extra_targets = set()
        return self._extra_targets

    @extra_targets.setter
    def extra_targets(self, value):
        assert isinstance(value, set)
        self._extra_targets = value

    def clone(self, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError('only Tensor can be cloned.')
        self._name = tensor.name
        self.expressions = tensor.expressions
        self._grad_wrts = tensor.grad_wrts
        self.extra_targets = tensor.extra_targets
        self.shape = tensor.shape

# ------------------------  Overloaded Operators  ------------------------

    def __str__(self):
        def_name = self.name
        real_name = ws.GetTensorName(def_name)
        shape = ', '.join(str(dim) for dim in self.shape) if self.shape is not None else None
        if shape is not None:
            return 'Tensor | name: {}, shape: {} '.format(def_name, '[' + shape + ']')
        else: return 'Tensor | name: {}'.format(def_name)

    def __getattr__(self, op_type):
        if not op_type in self.REGISTERED_FILLERS:
            if not op_type in config.REGISTERED_OPERATORS:
                return super(Tensor, self).__getattribute__(op_type)

        def wrapper(self, op_type, **kwargs):
            if op_type in self.REGISTERED_FILLERS:
                return self.Fill(op_type, **kwargs)
            elif op_type in config.REGISTERED_OPERATORS:
                return self.CreateOperator(inputs=[self], nout=1, op_type=op_type, **kwargs)

        return lambda self=self, op_type=op_type, **kwargs: wrapper(self, op_type, **kwargs)

    def __getitem__(self, item):
        def wrapper_indices(indices):
            tensor = Tensor(GetTensorName())
            ws.FeedTensor(tensor, np.array(indices, dtype=np.float32))
            return tensor

        if isinstance(item, int):
            output = self.CreateOperator(inputs=[self, wrapper_indices([item])], nout=1, op_type='At')
            if self.shape is not None:
                output.shape = self.shape[:]
                output.shape[0] = 1
            return output

        elif isinstance(item, slice):
            indices = [i for i in xrange(item.start, item.stop, item.step
                                         if item.step is not None else 1)]
            outputs = []
            for idx in indices:
                output = self.CreateOperator(inputs=[self, wrapper_indices([idx])], nout=1, op_type='At')
                if self.shape is not None:
                    output.shape = self.shape[:]
                    output.shape[0] = 1
                outputs.append(output)
            return outputs

        elif isinstance(item, Tensor):
            return self.CreateOperator(inputs=[self, item], nout=1, op_type='At')

    def __add__(self, other):
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                if not isinstance(other, list): other = [other]
            other = np.array(other, dtype=np.float32)
            tensor = Tensor(GetTensorName())
            ws.FeedTensor(tensor, other)
            other = tensor
        output = self.CreateOperator(inputs=[self, other], nout=1, op_type= 'Add')
        if self.shape is not None:
            output.shape = self.shape[:]
        return output

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                if not isinstance(other, list): other = [other]
            other = np.array(other, dtype=np.float32)
            tensor = Tensor(GetTensorName())
            ws.FeedTensor(tensor, other)
            other = tensor
        output = self.CreateOperator(inputs=[self, other], nout=1, op_type='Sub')
        if self.shape is not None:
            output.shape = self.shape[:]
        return output

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                if not isinstance(other, list): other = [other]
            other = np.array(other, dtype=np.float32)
            tensor = Tensor(GetTensorName())
            ws.FeedTensor(tensor, other)
            other = tensor
        output = self.CreateOperator(inputs=[self, other], nout=1, op_type='Mul')
        if self.shape is not None:
            output.shape = self.shape[:]
        return output

    def __div__(self, other):
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                if not isinstance(other, list): other = [other]
            other = np.array(other, dtype=np.float32)
            tensor = Tensor(GetTensorName())
            ws.FeedTensor(tensor, other)
            other = tensor
        output = self.CreateOperator(inputs=[self, other], nout=1, op_type='Div')
        if self.shape is not None:
            output.shape = self.shape[:]
        return output

    def __neg__(self):
        return self.__mul__(-1.0)

# ------------------------  Theano APIs  ------------------------

    def get_value(self):
        return ws.FetchTensor(self)

    def reshape(self, shape, **kwargs):
        if not isinstance(shape, tuple) \
                and not isinstance(shape, list): shape = [shape]
        return Tensor.CreateOperator(inputs=[self], nout=1, op_type='Reshape',
                                     shape=shape, **kwargs)

# ------------------------  TensorFlow APIs  ------------------------

    def get_shape(self):

        class TensorShape(object):

            class Dimension(object):
                def __init__(self, dim):
                    self.dim = dim
                def __str__(self):
                    return 'Dimension({})'.format(self.dim)

            def __init__(self, shape):
                self.dims = [self.Dimension(dim) for dim in shape]
                self.shape = shape

            def __str__(self):
                dims = [str(dim) for dim in self.dims]
                return 'TensorShape([{}])'.format(', '.join(dims))

            def as_list(self):
                return self.shape

        return TensorShape(self.shape)

# ------------------------  Utils  ------------------------

    @classmethod
    def CreateOperator(cls, inputs, op_type, nout=None, existing_outputs=None,
                       extra_inputs=None, name=None, **kwargs):

        expressions = OrderedDict()  # keep order for displaying

        # 1. collect inputs
        if not isinstance(inputs, list): inputs = [inputs]
        for input in inputs:
            for op_idx, expr in input.expressions.items():
                if not op_idx in expressions:
                    expressions[op_idx] = expr

        if extra_inputs is not None:
            if not isinstance(extra_inputs, list): extra_inputs = [extra_inputs]
            for input in extra_inputs:
                for op_idx, expr in input.expressions.items():
                    if not op_idx in expressions:
                        expressions[op_idx] = expr

        # 2. generate outputs
        outputs = []
        if existing_outputs is None:
            for idx in xrange(nout): outputs.append(Tensor())
        else:
            if not isinstance(existing_outputs, list): existing_outputs = [existing_outputs]
            outputs = existing_outputs
            nout = len(outputs)
            if not isinstance(outputs, list): outputs = [outputs]

        # 3. make def, then push back to expressions
        inputs_name = [input.name for input in inputs]
        outputs_name = [output.name for output in outputs]
        op_idx, op_name = GetOperatorName(name)
        device_option = None
        from dragon.core.scope import DEVICE_SCOPE, ENGINE_SCOPE
        if DEVICE_SCOPE != '':
            supports = {'/cpu': 0, '/gpu': 1}
            device_option = pb.DeviceOption()
            device_option.device_type = supports[DEVICE_SCOPE.split(':')[0]]
            device_option.gpu_id = int(DEVICE_SCOPE.split(':')[1])
            device_option.engine = ENGINE_SCOPE
        op_def = MakeOperatorDef(op_type, inputs_name, outputs_name, op_name,
                                 device_option=device_option, **kwargs)
        expressions[op_idx] = op_def

        # 4. deliver expression & extra_targets to all outputs
        for output in outputs:
            output.expressions = expressions
            # deliver extra_targets if necessary
            for input in inputs:
                output.extra_targets = \
                    output.extra_targets.union(input.extra_targets)
            if extra_inputs is not None:
                for input in extra_inputs:
                    output.extra_targets.add(input.name)

        # 5. utils
        if 'static_shape' in kwargs:
            outputs[0].tf_shape = kwargs['static_shape']

        if nout > 1:
            return outputs
        elif nout == 1:
            return outputs[0]
        else:
            return None

    def Fill(self, type, **kwargs):
        filler = pb.TensorFiller()
        filler.tensor = self._name
        filler.type = type.lower()

        if filler.type == 'constant':
            filler.value = kwargs['value'] if 'value' in kwargs else 0
        elif filler.type == 'normal' or filler.type == 'gaussian':
            filler.mean = kwargs['mean'] if 'mean' in kwargs else 0
            filler.std = kwargs['std'] if 'std' in kwargs else 1
            filler.type = 'normal'
        elif filler.type == 'uniform':
            filler.low = kwargs['low'] if 'low' in kwargs else 0
            filler.high = kwargs['high'] if 'high' in kwargs else 1
            filler.type = 'uniform'
        elif filler.type == 'truncated_normal' or filler.type == 'truncatednormal':
            filler.mean = kwargs['mean'] if 'mean' in kwargs else 0
            filler.std = kwargs['std'] if 'std' in kwargs else 1
            filler.low = filler.mean - 2.0 * filler.std
            filler.high = filler.mean + 2.0 * filler.std
            filler.type = 'truncated_normal'
        elif filler.type == 'parameterized_truncated_normal':
            filler.mean = kwargs['mean'] if 'mean' in kwargs else 0
            filler.std = kwargs['std'] if 'std' in kwargs else 1
            filler.low = kwargs['low'] if 'low' in kwargs else -2.0
            filler.high = kwargs['high'] if 'high' in kwargs else 2.0
        ws.CreateFiller(filler)
        return self

    def PrintExpressions(self):
        external_inputs = set()
        outputs = set()
        buffer0 = '-------------------Expressions-------------------\n'
        buffer1 = ''; buffer2 = 'Inputs: ['

        for k,v in self.expressions.items():
            buffer1 = buffer1 + '>>>  ' + str(k).zfill(3) + '. ('
            for input in v.input:
                if input not in outputs:
                    external_inputs.add(input)
                buffer1 = buffer1 + input + ', '
            buffer1 = buffer1 + 'None, ' if len(v.input) == 0 else buffer1
            buffer1 = buffer1[0:-2] + ') -> ' + v.type + ' -> ('
            for output in v.output:
                outputs.add(output)
                buffer1 = buffer1 + output + ', '
            buffer1 = buffer1[0:-2] + ') \n'

        buffer1 = buffer1 + 'Target: ' + self._name + '\n'
        for ex_input in external_inputs:
            buffer2 = buffer2 + ex_input + ', '
        buffer2 = buffer2 + ']\n'
        return buffer0 + buffer2 + buffer1 + buffer0

    def __call__(self, *args, **kwargs):
       return self.PrintExpressions()