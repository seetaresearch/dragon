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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import dragon.core.workspace as ws
import dragon.protos.dragon_pb2 as pb
from dragon.core.utils import MakeOperatorDef
from dragon.core.scope import GetOperatorName, GetTensorName


class Tensor(object):
    """
    Tensor is generally used to represent a n-dim array,
    across all virtual frontends, with a optional name and shape.

    It does not have any storage, i.e., just provided to be a navigation
    to the real tensor in the C++ backend.
    """
    def __init__(self, name=None, shape=None, dtype=None, _name=None):
        """Construct a Tensor instance.

        Parameters
        ----------
        name : None or str
            The name of Tensor.
        shape : None or list
            The shape of Tensor.
        dtype : None or str
            The type of Tensor.
        _name : None or str
            The name that ignores name scope.

        Returns
        -------
        Tensor
            An unregistered Tensor.

        """
        if _name is not None: self._name = _name
        else: self.name = name
        self.shape = shape
        self.dtype = dtype

    ##############################################
    #                                            #
    #                 REGISTERS                  #
    #                                            #
    ##############################################

    def _no_parameter_filler(self, type):
        filler = pb.TensorFiller()
        filler.tensor = self.name
        filler.type = type
        ws.CreateFiller(filler)
        return self

    def Variable(self):
        """
        Register as an empty variable.
        """
        ws.CreateTensor(self.name)
        return self

    def Placeholder(self):
        """
        Register as a placeholder.
        """
        ws.CreateTensor(self.name)
        return self

    def Constant(self, value=0):
        """Register as a variable with constant initializer.

        Parameters
        ----------
        value : basic numerical type
            The constant value.

        """
        filler = pb.TensorFiller()
        filler.tensor = self.name
        filler.type = 'constant'
        filler.value = value
        ws.CreateFiller(filler)
        return self

    def Uniform(self, low=-1, high=1):
        """Register as a variable with uniform initializer.

        Parameters
        ----------
        low : basic numerical type
             The lower bound of uniform distribution.
        high : basic numerical type
            The higher bound of uniform distribution.

        """
        filler = pb.TensorFiller()
        filler.tensor = self.name
        filler.type = 'uniform'
        filler.low = low
        filler.high = high
        ws.CreateFiller(filler)
        return self

    def Normal(self, mu=0, sigma=1):
        """Register as a variable with normal initializer.

        Parameters
        ----------
        mu : basic numerical type
            The mu of normal distribution.
        sigma : basic numerical type
            The sigma of normal distribution.

        """
        filler = pb.TensorFiller()
        filler.tensor = self.name
        filler.type = 'normal'
        filler.mean= mu
        filler.std = sigma
        ws.CreateFiller(filler)
        return self

    def TruncatedNormal(self, mu=0, sigma=1):
        """Register as a variable with truncated normal initializer.

        Parameters
        ----------
        mu : basic numerical type
            The mu of normal distribution.
        sigma : basic numerical type
            The sigma of normal distribution.

        """
        filler = pb.TensorFiller()
        filler.tensor = self.name
        filler.type = 'truncated_normal'
        filler.mean = mu
        filler.std = sigma
        filler.low = mu - 2.0 * sigma
        filler.high = mu + 2.0 * sigma
        ws.CreateFiller(filler)
        return self

    def Gaussian(self, mean=0, std=1):
        """Register as a variable with gaussian initializer.

        Parameters
        ----------
        mean : basic numerical type
            The mean(mu) of normal distribution.
        std : basic numerical type
            The std(sigma) of normal distribution.

        """
        return self.Normal(mu=mean, sigma=std)

    def Xavier(self, scale=3.0):
        """
        Register as a variable with xavier initializer.
        """
        filler = pb.TensorFiller()
        filler.tensor = self.name
        filler.type = 'xavier'
        filler.scale = scale
        ws.CreateFiller(filler)
        return self

    def MSRA(self, scale=2.0):
        """
        Register as a variable with msra initializer.
        """
        filler = pb.TensorFiller()
        filler.tensor = self.name
        filler.type = 'msra'
        filler.scale = scale
        ws.CreateFiller(filler)
        return self

    def GlorotUniform(self, scale=3.0):
        """
        Register as a variable with glorot uniform initializer.
        """
        return self.Xavier(scale)

    def GlorotNormal(self, scale=2.0):
        """
        Register as a variable with glorot normal initializer.
        """
        return self.MSRA(scale)

    ##############################################
    #                                            #
    #                 PROPERTIES                 #
    #                                            #
    ##############################################

    @property
    def expressions(self):
        """Return or Set the expressions.

        Parameters
        ----------
        value : dict
            The expressions to be used.

        Returns
        -------
        dict
            The internal expressions that it has currently stored.

        """
        if not hasattr(self,'_expressions'): self._expressions = {}
        return self._expressions

    @expressions.setter
    def expressions(self, value):
        assert isinstance(value, dict)
        self._expressions = value

    @property
    def name(self):
        """Return or Set the name.

        Parameters
        ----------
        value : None or str
            The name to set.

        Returns
        -------
        str
            The name of this tensor.

        """
        return self._name

    @name.setter
    def name(self, value):
        from .scope import _TENSOR_SCOPE
        if value is None:
            # ignore the scope for the name generated by uid
            self._name = GetTensorName()
        else:
            self._name = _TENSOR_SCOPE + value

    @property
    def grad_wrts(self):
        """Return or Set the gradients w.r.t.

        Returns
        -------
        list
            The list of names represents gradients w.r.t.

        """
        if not hasattr(self,'_grad_wrts'): self._grad_wrts = []
        return self._grad_wrts

    @property
    def grad_objs(self):
        """Return or Set the gradients objectives.

        Returns
        -------
        list
             The list of names represents objectives.

        """
        if not hasattr(self, '_grad_objs'): self._grad_objs = []
        return self._grad_objs

    @property
    def shape(self):
        """Return or Set the shape.

        Returns
        -------
        list or None
            The shape of this tensor.

        """
        if not hasattr(self, '_shape'): self._shape = None
        return self._shape

    @shape.setter
    def shape(self, value):
        if value is not None:
            if not isinstance(value, (tuple, list)):
                raise TypeError('The shape should be a tuple or list.')
            self._shape = list(value)
        else: self._shape = value

    @property
    def ndim(self):
        """Return the ndim.

        If shape is ``None``, returns ``0`` instead.

        Returns
        -------
        int
            The ndim.

        """
        if self._shape is not None: return len(self._shape)
        else: return -1

    @property
    def dtype(self):
        """Return or Set the data type.

        Parameters
        ----------
        value : str or None
            The data type to set.

        Returns
        -------
        str or None
            The data type.

        """
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    def astype(self, dtype, inplace=False):
        """Cast the data type of inputs to a specific one.

        Parameters
        ----------
        dtype : str
            The specific dtype.
        inplace : boolean
            Whether to modify the inputs.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if inplace:
            output = Tensor.CreateOperator(inputs=[], existing_outputs=[self],
                op_type='AsType', dtype=dtype)
        else:
            output = Tensor.CreateOperator(inputs=self, nout=1,
                op_type='AsType', dtype=dtype)

        if self.shape is not None:
            output.shape = self.shape[:]

        return output

    @property
    def extra_targets(self):
        """Return or Set the extra solving targets.

        Parameters
        ----------
        value : set
            Set the extra solving targets.

        Returns
        -------
        set
            The extra solving targets.

        """
        if not hasattr(self,'_extra_targets'): self._extra_targets = set()
        return self._extra_targets

    @extra_targets.setter
    def extra_targets(self, value):
        assert isinstance(value, set)
        self._extra_targets = value

    def clone(self, tensor):
        """Clone a existing tensor.

        Parameters
        ----------
        tensor : Tensor
            Set the extra solving targets.

        Returns
        -------
        None

        """
        if not isinstance(tensor, Tensor):
            raise TypeError('Only Tensor can be cloned.')
        self.__dict__ = tensor.__dict__

    #################################################
    #                                               #
    #                   OVERRIDES                   #
    #                                               #
    #################################################

    def __repr__(self):
        """Return a format str representing the tensor.

        Returns
        -------
        str
            The format str.

        """
        def_name = self.name
        real_name = ws.GetTensorName(def_name)
        shape = ', '.join(str(dim) for dim in self.shape) if self.shape else None
        if shape:
            return '[dragon.Tensor({} -> {}) of shape {}]'\
                .format(def_name, real_name, '(' + shape + ')')
        else:
            return '[dragon.Tensor({} -> {})]'.format(def_name, real_name)

    def __getitem__(self, item):
        """Return a Tensor with specific indices.

        Parameters
        ----------
        item : int, slice or Tensor
            The indices.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(item, (slice, tuple)):
            if not isinstance(item, int):
                raise ValueError('The index should be a integer.')
            item = (item,)
        if not isinstance(item, tuple): item = tuple([item])
        starts = []; ends = []; output_dims = []
        for ix, it in enumerate(item):
            if isinstance(it, slice):
                # handle start
                if it.start is None: starts.append(0)
                else: starts.append(it.start)
                # handle stop
                if it.stop is None:
                    ends.append(0)
                    output_dims.append(None)
                else:
                    ends.append(it.stop)
                    output_dims.append(ends[-1] - starts[-1])
                    if starts[-1] == ends[-1]:
                        raise ValueError('The cropping starts and ends of axis {} '
                                         'can not be equal, got {}:{}.'
                                         .format(ix, starts[-1], ends[-1]))
                # handle step
                if it.step is not None:
                    raise NotImplementedError('Cropping with step has not been implemented yet. ')
            elif isinstance(it, int):
                starts.append(it)
                ends.append(-1)
                output_dims.append(-1)
            else:
                raise TypeError('Unsupported type of indices: {}'.format(type(type(it))))

        output = self.CreateOperator(inputs=self, nout=1, op_type='Crop', starts=starts, ends=ends)

        if self.shape is not None:
            output_shape = self.shape[:]
            squeeze_shape = []
            for ix in range(len(output_dims)):
                output_shape[ix] = output_dims[ix]
            for dim in output_shape:
                if dim != -1: squeeze_shape.append(dim)
            if len(squeeze_shape) == 0: output.shape = None
            else: output.shape = squeeze_shape[:]

        return output

    def __add__(self, other):
        """Calculate x + y.

        Parameters
        ----------
        other : Tensor
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
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

    def __radd__(self, other):
        """Calculate y + x.

        Parameters
        ----------
        other : Tensor
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                if not isinstance(other, list): other = [other]
            other = np.array(other, dtype=np.float32)
            tensor = Tensor(GetTensorName())
            ws.FeedTensor(tensor, other)
            other = tensor
        output = self.CreateOperator(inputs=[other, self], nout=1, op_type= 'RAdd')
        if self.shape is not None:
            output.shape = self.shape[:]
        return output

    def __sub__(self, other):
        """Calculate x - y.

        Parameters
        ----------
        other : Tensor
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
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

    def __rsub__(self, other):
        """Calculate y - x.

        Parameters
        ----------
        other : Tensor
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                if not isinstance(other, list): other = [other]
            other = np.array(other, dtype=np.float32)
            tensor = Tensor(GetTensorName())
            ws.FeedTensor(tensor, other)
            other = tensor
        output = self.CreateOperator(inputs=[other, self], nout=1, op_type='RSub')
        if self.shape is not None:
            output.shape = self.shape[:]
        return output

    def __mul__(self, other):
        """Calculate x * y.

        Parameters
        ----------
        other : Tensor
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
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

    def __rmul__(self, other):
        """Calculate y * x.

        Parameters
        ----------
        other : Tensor
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                if not isinstance(other, list): other = [other]
            other = np.array(other, dtype=np.float32)
            tensor = Tensor(GetTensorName())
            ws.FeedTensor(tensor, other)
            other = tensor
        output = self.CreateOperator(inputs=[other, self], nout=1, op_type='RMul')
        if self.shape is not None:
            output.shape = self.shape[:]
        return output

    def __div__(self, other):
        """Calculate x / y.

        Parameters
        ----------
        other : Tensor
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
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

    __truediv__ = __div__

    def __rdiv__(self, other):
        """Calculate y / x.

        Parameters
        ----------
        other : Tensor
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                if not isinstance(other, list): other = [other]
            other = np.array(other, dtype=np.float32)
            tensor = Tensor(GetTensorName())
            ws.FeedTensor(tensor, other)
            other = tensor
        output = self.CreateOperator(inputs=[other, self], nout=1, op_type='RDiv')
        if self.shape is not None:
            output.shape = self.shape[:]
        return output

    __rtruediv__ = __rdiv__

    def __neg__(self):
        """Calculate -x.

        Returns
        -------
        Tensor
            The output tensor.

        """
        return self.__mul__(-1.0)

    def __call__(self, *args, **kwargs):
        """Print the expressions.

        Returns
        -------
        None

        """
        return self.debug_expressions()

    ###############################################
    #                                             #
    #                 THEANO-APIS                 #
    #                                             #
    ###############################################

    def set_value(self, new_value, **kwargs):
        """Feed the values to C++ backend. [**Theano Style**]

        Parameters
        ----------
        new_value : number, list or numpy.ndarray
            The values to set.

        Returns
        -------
        None

        See Also
        --------
        `workspace.FeedTensor(*args, **kwargs)`_ - How to feed a Tensor.

        """
        ws.FeedTensor(self, new_value)

    def get_value(self):
        """Fetch the values from C++ backend. [**Theano Style**]

        Returns
        -------
        numpy.ndarray
            The values of this tensor in the backend.

        See Also
        --------
        `workspace.FetchTensor(*args, **kwargs)`_ - How to fetch a Tensor.

        """
        return ws.FetchTensor(self)

    def copy(self):
        """Return a Tensor with same content. [**Theano Style**]

        Returns
        -------
        Tensor
            The copy.

        See Also
        --------
        `ops.Copy(*args, **kwargs)`_ - How to copy A to B.

        """
        new_tensor = Tensor(self.name + '_copy')
        arguments = {'inputs': self,
                     'existing_outputs': new_tensor}
        self.CreateOperator(nout=1, op_type='Copy', **arguments)

        if self.shape is not None:
            new_tensor.shape = self.shape[:]

        return new_tensor

    def reshape(self, shape, **kwargs):
        """Reshape the dimensions of input. [**Theano Style**]

        Parameters
        ----------
        shape : tuple or list
            The new shape.

        Returns
        -------
        Tensor
            The reshaped output.

        """
        if not isinstance(shape, tuple) \
                and not isinstance(shape, list): shape = [shape]
        output = Tensor.CreateOperator(inputs=self, nout=1, op_type='Reshape',
                                       shape=shape, **kwargs)
        if self.shape is not None:
            output.shape = list(shape)
        return output

    def dimshuffle(self, *args, **kwargs):
        """Shuffle the dimensions. [**Theano Style**]

        Parameters
        ----------
        dimensions : list
            The desired dimensions.

        Returns
        -------
        Tensor
            The dimshuffled output.

        """
        dimensions = list(args)
        perms = []
        for dim in dimensions:
            if dim != 'x':
                if not isinstance(dim, int):
                    raise ValueError('The type of dimension should be int.')
                perms.append(dim)

        # transpose
        output = Tensor.CreateOperator(inputs=self, nout=1,
                                       op_type='Transpose', perms=perms, **kwargs)
        if self.shape is not None:
            if len(self.shape) != len(perms):
                raise ValueError('The ndim of inputs is {}, but perms provide {}'. \
                                 format(len(self.shape), len(perms)))
            output.shape = self.shape[:]
            for i, axis in enumerate(perms):
                output.shape[i] = self.shape[axis]

        # expand dims
        for i in range(len(dimensions) - len(perms)):
            flag = False
            input_shape = output.shape
            axis = -1
            for idx in range(len(dimensions)):
                if idx >= len(perms): continue
                cur_dim = perms[idx]; exp_dim = dimensions[idx]
                if cur_dim != exp_dim:
                    axis = idx
                    output = Tensor.CreateOperator(inputs=output, nout=1,
                                    op_type='ExpandDims', axis=axis)
                    perms.insert(axis, 'x')
                    flag = True
                    break
            if not flag:
                axis = len(perms)
                output = Tensor.CreateOperator(inputs=output, nout=1,
                                    op_type='ExpandDims', axis=len(perms))
                perms.append('x')

            if self.shape is not None:
                output.shape = input_shape[:]
                output.shape.insert(axis, np.long(1))

        return output

    ########################################################
    #                                                      #
    #                   TENSORFLOW-APIS                    #
    #                                                      #
    ########################################################

    def get_shape(self):
        """Construct the shape descriptor. [**TensorFlow Style**]

        Returns
        -------
        TensorShape or None
            The shape description.

        Examples
        --------
        >>> a = Tensor(shape=[1, 2, 3, 4])
        >>> print a.get_shape()
        >>> TensorShape([Dimension(1), Dimension(2), Dimension(3), Dimension(4)])

        >>> print a.get_shape().as_list()
        >>> [1, 2, 3, 4]

        """
        raise NotImplementedError('Implemented in <vm.tensorflow.framework.tensor_shape>')

    def eval(self, feed_dict=None):
        """Run and return the computing results of this tensor.

        Parameters
        ----------
        feed_dict : dict
            The values to feed.

        Returns
        -------
        numpy.ndarray
            The values of this tensor in the backend.
        """
        raise NotImplementedError('Implemented in <vm.theano.compile.function>')

    ############################################
    #                                          #
    #                   MISC                   #
    #                                          #
    ############################################

    @classmethod
    def CreateOperator(cls, inputs, op_type,
                       nout=None, existing_outputs=None, output_shapes=None,
                       extra_inputs=None, name=None, **kwargs):
        """Construct a new Tensor with specific operator descriptor.

        Parameters
        ----------
        inputs : list of Tensor or Tensor
            The inputs for this operator.
        op_type : str
            The operator type.
        nout : int
            The number of outputs to return.
            It will be discarded if ``existing_outputs`` is not None.
        existing_outputs : list of Tensor, Tensor or None
            The existing outputs for this operator.
        extra_inputs : list of Tensor, Tensor or None
            The inputs that should be attached to solving targets, e.g. dynamic shape.
        name : str or None
            The optional name to use. ``Op_xxx`` will be used automatically if it is None.

        Returns
        -------
        list of Tensor, Tensor, or None
            The outputs of this operator.

        Examples
        --------
        >>> import dragon as dg
        >>> a = Tensor().Variable()
        >>> b = Tensor().Variable()
        >>> c = Tensor.CreateOperator(inputs=[a, b], op_type='Add', nout=1)

        >>> a = Tensor().Variable()
        >>> b = Tensor().Variable()
        >>> c = Tensor().Variable()
        >>> c = Tensor.CreateOperator(inputs=[a, b], op_type='Add', existing_outputs=c)

        >>> dynamic_shape = Tensor().Variable()
        >>> dg.workspace.FeedTensor(dynamic_shape, [1, 2, 3, 4])
        >>> a = dg.Fill(shape=dynamic_shape, value=5.0)
        >>> print dg.function(outputs=a)
        >>> [[ 5.  5.  5.]
             [ 5.  5.  5.]]

        """
        expressions = dict()

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
            for idx in range(nout): outputs.append(Tensor())
        else:
            if not isinstance(existing_outputs, list): existing_outputs = [existing_outputs]
            outputs = existing_outputs
            nout = len(outputs)
            if not isinstance(outputs, list): outputs = [outputs]

        # 3. make def, then push expressions
        inputs_name = [input.name for input in inputs]
        outputs_name = [output.name for output in outputs]
        op_idx, op_name = GetOperatorName(name)
        device_option = None
        from dragon.core.scope import _DEVICE_SCOPE, _ENGINE_SCOPE
        if _DEVICE_SCOPE != '':
            supports = {'/cpu': 0, '/gpu': 1}
            device_option = pb.DeviceOption()
            device_option.device_type = supports[_DEVICE_SCOPE.split(':')[0]]
            device_option.device_id = int(_DEVICE_SCOPE.split(':')[1])
            device_option.engine = _ENGINE_SCOPE
        op_def = MakeOperatorDef(op_type, inputs_name, outputs_name, op_name,
                                 device_option=device_option, **kwargs)
        expressions[op_idx] = op_def

        # 4. make outputs
        for idx, output in enumerate(outputs):
            # deliver expressions
            output.expressions = expressions
            # deliver extra targets
            for input in inputs:
                output.extra_targets = \
                    output.extra_targets.union(input.extra_targets)
            if extra_inputs is not None:
                for input in extra_inputs:
                    output.extra_targets.add(input.name)

        # 5. utils
        if 'static_shape' in kwargs:
            outputs[0].tf_shape = kwargs['static_shape']

        # 6. returns
        if nout > 1: return outputs
        elif nout == 1: return outputs[0]
        else: return None

    @classmethod
    def Convert(cls, value, dtype='float32'):
        """Convert the given value to a tensor.

        Parameters
        ----------
        value : numerical type
            The value to convert.
        dtype : str
            The data type of the tensor.

        Returns
        -------
        Tensor
            The tensor converted with given value.

        """
        if isinstance(value, Tensor):
            return value
        else:
            if isinstance(value, (list, tuple)):
                np_value = np.array(value, dtype=dtype)
            elif isinstance(value, np.ndarray):
                np_value = value.astype(dtype=dtype)
            else:
                try:
                    np_value = np.array(value, dtype=dtype)
                except:
                    raise TypeError('{} value can not be converted to tensor.'.format(type(value)))
            tensor = Tensor(shape=list(np_value.shape), dtype=dtype)
            tensor.set_value(np_value)
            return tensor

    def Fill(self, type, **kwargs):
        """Fill self with the specific type of filler.

        Parameters
        ----------
        type : str
            The type of the filler.

        Returns
        -------
        Tensor
            Self, with filler registered implicitly in the backend.

        """
        filler = pb.TensorFiller()
        filler.tensor = self.name
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

    def debug_expressions(self):
        """Return the internal expressions for displaying.

        Returns
        -------
        str
            The internal expressions.

        """
        external_inputs = set()
        outputs = set()
        ordered_exprs = sorted(self.expressions.items(), key=lambda d: d[0])
        buffer0 = '-------------------Expressions-------------------\n'
        buffer1 = ''; buffer2 = 'Inputs: ['

        for k, v in ordered_exprs:
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