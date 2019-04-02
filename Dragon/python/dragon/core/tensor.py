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

"""Center of the Virtual Stack.

We will reuse this ``Tensor`` structure across all frameworks,
whose computation graph is static.

For the dynamic computation graph case, see ``vm.torch.Tensor``.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core import scope as _scope
from dragon.core import helper as _helper
from dragon.core import workspace as _workspace
from dragon.proto import dragon_pb2 as _proto_def
from dragon.core import proto_utils as _proto_utils


class Tensor(object):
    """Tensor is generally used to represent a n-dimension array,
    across all virtual frameworks, with a optional name and shape.

    It does not have any storage, i.e., just provided to be a navigation
    to the real tensor in the C++ backend.

    """
    def __init__(self, name=None, shape=None, dtype=None):
        """Construct a Tensor instance.

        Parameters
        ----------
        name : str, optional
            The name of Tensor.
        shape : list, optional
            The shape of Tensor.
        dtype : str, optional
            The type of Tensor.

        Returns
        -------
        Tensor
            An unregistered Tensor.

        """
        self.name, self.shape, self.dtype = name, shape, dtype
        self.gradient = _helper.GradientHelper(self)

    ##############################################
    #                                            #
    #                 Registers                  #
    #                                            #
    ##############################################

    def Variable(self):
        """Register as an empty variable.

        Returns
        -------
        Tensor
            The self.

        """
        return self.Fill('variable')

    def Placeholder(self):
        """Register as a placeholder.

        Returns
        -------
        Tensor
            The self.

        """
        return self.Fill('placeholder')

    def Constant(self, value=0):
        """Register as a variable with constant initializer.

        Parameters
        ----------
        value : number, optional, default=0
            The constant value.

        Returns
        -------
        Tensor
            The self.

        """
        return self.Fill('constant', value=value)

    def Uniform(self, low=0, high=1):
        """Register as a variable with uniform initializer.

        Parameters
        ----------
        low : number, optional, default=0
             The lower bound of uniform distribution.
        high : number, optional, default=1
            The higher bound of uniform distribution.

        Returns
        -------
        Tensor
            The self.

        """
        return self.Fill('uniform', low=low, high=high)

    def Normal(self, mu=0, sigma=1):
        """Register as a variable with normal initializer.

        Parameters
        ----------
        mu : number, optional, default=0
            The mu of normal distribution.
        sigma : number, optional, default=1
            The sigma of normal distribution.

        Returns
        -------
        Tensor
            The self.

        """
        return self.Fill('normal', mean=mu, std=sigma)

    def TruncatedNormal(self, mu=0, sigma=1):
        """Register as a variable with truncated normal initializer.

        Parameters
        ----------
        mu : number, optional, default=0
            The mu of normal distribution.
        sigma : number, optional, default=1
            The sigma of normal distribution.

        Returns
        -------
        Tensor
            The self.

        """
        return self.Fill('truncated_normal', mean=mu, std=sigma)

    def Gaussian(self, mean=0, std=1):
        """Register as a variable with gaussian initializer.

        Parameters
        ----------
        mean : number, optional, default=0
            The mean(mu) of normal distribution.
        std : number, optional, default=1
            The std(sigma) of normal distribution.

        Returns
        -------
        Tensor
            The self.

        """
        return self.Normal(mu=mean, sigma=std)

    def GlorotUniform(self, scale=3.):
        """Register as a variable with glorot uniform initializer.

        Parameters
        ----------
        scale : number, optional, default=3.
            The scale factor.

        Returns
        -------
        Tensor
            The self.

        """
        return self.Fill('glorot_uniform', scale=scale)

    def GlorotNormal(self, scale=2.):
        """Register as a variable with glorot normal initializer.

        Parameters
        ----------
        scale : number, optional, default=2.
            The scale factor.

        Returns
        -------
        Tensor
            The self.

        """
        return self.Fill('glorot_normal', scale=scale)

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
        value : str
            The name to set.

        Returns
        -------
        str
            The name of this tensor.

        """
        return self._name

    @name.setter
    def name(self, value):
        if value != '':
            self._name = _workspace.GetDummyName(
                _scope.get_default_name_scope() + value
                    if value else 'Tensor', domain='Tensor')
        else:
            # Set it manually for same cases
            self._name = value

    def set_name(self, name):
        """Set the name while getting rid of name scope.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        None

        """
        self._name = name

    @property
    def shape(self):
        """Return or Set the shape.

        Parameters
        ---------
        value : sequence of int
            The shape to set.

        Returns
        -------
        sequence of int
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
        value : str
            The data type to set.

        Returns
        -------
        str
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
        inplace : boolean, optional, default=False
            Whether to modify the inputs.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if inplace:
            return Tensor.CreateOperator(
                'Cast', [], existing_outputs=[self], dtype=dtype)
        else:
            return Tensor.CreateOperator('Cast', self, dtype=dtype)

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
    #                   Overrides                   #
    #                                               #
    #################################################

    def __repr__(self):
        """Return a format str representing the tensor.

        Returns
        -------
        str
            The format str.

        """
        shape_str = ('(' + ', '.join(['?' if str(dim) == 'None' else str(dim)
            for dim in self.shape]) + (',)' if len(self.shape) == 1 else ')')) \
                if self.shape is not None else 'None'
        return 'Tensor("{}", shape={}, dtype={})' \
            .format(self.name, shape_str, self.dtype)

    def _process_indices(self, item):
        if not isinstance(item, (slice, tuple)):
            if not isinstance(item, int):
                raise ValueError('The index should be a integer.')
            item = (item,)
        if not isinstance(item, tuple): item = tuple([item])
        starts, sizes = [], []
        for ix, it in enumerate(item):
            if isinstance(it, slice):
                # Handle start
                if it.start is None: starts.append(0)
                else: starts.append(it.start)
                # Handle stop
                if it.stop is None:
                    sizes.append(-1)
                else:
                    sizes.append(it.stop - starts[-1])
                    if sizes[-1] == 0:
                        raise ValueError(
                            'The starts and ends of axis {} '
                                'can not be equal, got {}:{}.'
                                    .format(ix, starts[-1], it.stop))
                # Handle step
                if it.step is not None:
                    raise NotImplementedError(
                        'Indexing with step has not been implemented yet. ')
            elif isinstance(it, int):
                starts.append(it)
                sizes.append(0)
            else:
                raise TypeError('Unsupported type of indices: {}'.format(type(it)))
        return starts, sizes

    def __getitem__(self, item):
        """Return the value at the specific indices.

        Parameters
        ----------
        item : int or slice
            The indices.

        Returns
        -------
        Tensor
            The output tensor.

        """
        starts, sizes = self._process_indices(item)
        output = self.CreateOperator('Crop', self, starts=starts, sizes=sizes)
        if self.shape is not None:
            output_shape, squeeze_shape = self.shape[:], []
            for ix in range(len(sizes)):
                output_shape[ix] = sizes[ix]
            for dim in output_shape:
                if dim != -1: squeeze_shape.append(dim)
            if len(squeeze_shape) == 0: output.shape = []
            else: output.shape = squeeze_shape[:]
        return output

    def __setitem__(self, key, value):
        """Set the value at the specific indices.

        Parameters
        ----------
        key : int or slice
            The indices.
        value : Tensor, number or sequence
            The value.

        Returns
        -------
        None

        """
        starts, sizes = self._process_indices(key)
        if not isinstance(value, Tensor):
            value = self._from_constants(value)
        return self.CreateOperator('Assign', [value],
            existing_outputs=[self], starts=starts, sizes=sizes)

    def _from_constants(self, value):
        if not isinstance(value, numpy.ndarray):
            try:
                value = numpy.array(value, dtype=self.dtype
                    if self.dtype else 'float32')
            except:
                raise TypeError(
                    'Can not convert the value to Tensor or numpy array.')
        ref_tensor =  Tensor.Ref(
            name=_workspace.GetDummyName('Constant',
                domain='Tensor', zero_based=False),
                    shape=list(value.shape), dtype=str(value.dtype))
        ref_tensor.set_value(value)
        return ref_tensor

    def __add__(self, other):
        """Calculate x + y.

        Parameters
        ----------
        other : Tensor or number
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('Add', [self, other])

    def __radd__(self, other):
        """Calculate y + x.

        Parameters
        ----------
        other : Tensor or number
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('RAdd', [other, self])

    def __sub__(self, other):
        """Calculate x - y.

        Parameters
        ----------
        other : Tensor or number
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('Sub', [self, other])

    def __rsub__(self, other):
        """Calculate y - x.

        Parameters
        ----------
        other : Tensor or number
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('RSub', [other, self])

    def __mul__(self, other):
        """Calculate x * y.

        Parameters
        ----------
        other : Tensor or number
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('Mul', [self, other])

    def __rmul__(self, other):
        """Calculate y * x.

        Parameters
        ----------
        other : Tensor or number
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('RMul', [other, self])

    def __div__(self, other):
        """Calculate x / y.

        Parameters
        ----------
        other : Tensor or number
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('Div', [self, other])

    __truediv__ = __div__

    def __rdiv__(self, other):
        """Calculate y / x.

        Parameters
        ----------
        other : Tensor or number
            The y.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('RDiv', [other, self])

    __rtruediv__ = __rdiv__

    def __neg__(self):
        """Calculate -x.

        Returns
        -------
        Tensor
            The output tensor.

        """
        return self.__mul__(-1.0)

    def __gt__(self, other):
        """Compute *self* > *other* element-wise.

        Parameters
        ----------
        other : Tensor or number
            The other tensor.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('Compare', [self, other], operation='GT')

    def __ge__(self, other):
        """Compute *self* > *other* element-wise.

        Parameters
        ----------
        other : Tensor or number
            The other tensor.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('Compare', [self, other], operation='GE')

    def __lt__(self, other):
        """Compute *self* < *other* element-wise.

        Parameters
        ----------
        other : Tensor or number
            The other tensor.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('Compare', [self, other], operation='LT')

    def __le__(self, other):
        """Compute *self* <= *other* element-wise.

        Parameters
        ----------
        other : Tensor or number
            The other tensor.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('Compare', [self, other], operation='LE')

    def __eq__(self, other):
        """Compute *self* == *other* element-wise.

        Parameters
        ----------
        other : Tensor or number
            The other tensor.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if not isinstance(other, Tensor):
            other = self._from_constants(other)
        return self.CreateOperator('Compare', [self, other], operation='EQ')

    def __hash__(self):
        return id(self)

    def __call__(self, *args, **kwargs):
        """Print the expressions.

        Returns
        -------
        None

        """
        return self.debug_expressions()

    ###############################################
    #                                             #
    #                 Theano API                  #
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
        Tensor
            The self.

        See Also
        --------
        `workspace.FeedTensor(*args, **kwargs)`_ - How to feed a Tensor.

        """
        _workspace.FeedTensor(self, new_value)
        return self

    def get_value(self):
        """Fetch the values from C++ backend. [**Theano Style**]

        Returns
        -------
        numpy.ndarray or number
            The values of this tensor in the backend.

        See Also
        --------
        `workspace.FetchTensor(*args, **kwargs)`_ - How to fetch a Tensor.

        """
        return _workspace.FetchTensor(self)

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
        new_tensor = Tensor.Ref(self.name + '_copy')
        arguments = {'inputs': self, 'existing_outputs': new_tensor}
        return self.CreateOperator('Copy', **arguments)

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
        if not isinstance(shape, (tuple, list)): shape = [shape]
        return Tensor.CreateOperator(
            'Reshape', inputs=self, shape=shape, **kwargs)

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

        # Transpose
        output = Tensor.CreateOperator(
            'Transpose', self, perms=perms, **kwargs)

        # Expand dims
        for i in range(len(dimensions) - len(perms)):
            flag = False
            input_shape = output.shape
            axis = -1
            for idx in range(len(dimensions)):
                if idx >= len(perms): continue
                cur_dim = perms[idx]; exp_dim = dimensions[idx]
                if cur_dim != exp_dim:
                    axis = idx
                    output = Tensor.CreateOperator(
                        'ExpandDims', output, axis=axis)
                    perms.insert(axis, 'x')
                    flag = True
                    break
            if not flag:
                axis = len(perms)
                output = Tensor.CreateOperator(
                    'ExpandDims', output, axis=len(perms))
                perms.append('x')

            if self.shape is not None:
                output.shape = input_shape[:]
                output.shape.insert(axis, 1)

        return output

    ########################################################
    #                                                      #
    #                   TensorFlow API                     #
    #                                                      #
    ########################################################

    def get_shape(self):
        """Construct the shape descriptor. [**TensorFlow Style**]

        Returns
        -------
        TensorShape
            The shape description.

        """
        raise NotImplementedError('')

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
        raise NotImplementedError('')

    ############################################
    #                                          #
    #                   Misc                   #
    #                                          #
    ############################################

    @classmethod
    def Ref(cls, name, shape=None, dtype=None):
        """Create a reference tensor from a unknown name.

        It is useful to get named Tensor navigator anywhere.

        Parameters
        ----------
        name : str
            The name of Tensor.
        shape : None or list
            The shape of Tensor.
        dtype : None or str
            The type of Tensor.

        Returns
        -------
        Tensor
            The ref tensor

        """
        ref_tensor = Tensor('', shape=shape, dtype=dtype)
        ref_tensor._name = name
        return ref_tensor

    @classmethod
    def CreateOperator(
        cls,
        op_type,
        inputs,
        num_outputs=1,
        existing_outputs=None,
        extra_inputs=None,
        name=None,
        **kwargs
    ):
        """Construct a new Tensor with specific operator descriptor.

        Parameters
        ----------
        op_type : str
            The type of operator.
        inputs : sequence of Tensor
            The inputs for this operator.
        num_outputs : int, optional, default=1
            The number of outputs to return.
        existing_outputs : sequence of Tensor, optional
            The existing outputs for this operator.
        extra_inputs : sequence of Tensor, optional
            The inputs that should be attached to solving targets.
        name : str, optional
            The optional name.

        Returns
        -------
        sequence of Tensor
            The outputs of this operator.

        Examples
        --------
        >>> import dragon as dg
        >>> a = Tensor().Variable()
        >>> b = Tensor().Variable()
        >>> c = Tensor.CreateOperator(inputs=[a, b], op_type='Add')

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

        # 1. Collect inputs
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

        # 2. Generate outputs
        outputs = []
        if existing_outputs is None:
            name_scope = _scope.get_default_name_scope()
            for idx in range(num_outputs):
                outputs.append(Tensor.Ref(
                    _workspace.GetDummyName(name_scope +
                        (name if name else op_type),
                            suffix=':{}'.format(idx),
                                domain='Tensor')))
        else:
            if not isinstance(existing_outputs, list):
                existing_outputs = [existing_outputs]
            outputs = existing_outputs
            num_outputs = len(outputs)
            if not isinstance(outputs, list): outputs = [outputs]

        # 3. Construct OperatorDef
        inputs_name = [input.name for input in inputs]
        outputs_name = [output.name for output in outputs]
        op_idx, op_name = _helper.OperatorHelper.get_index_and_name()
        device_option = _proto_utils.GetDefaultDeviceOption()

        op_def = _proto_utils.MakeOperatorDef(op_type,
            inputs_name, outputs_name, op_name,
                device_option=device_option, **kwargs)

        expressions[op_idx] = op_def

        # 4. Add outputs
        for idx, output in enumerate(outputs):
            # Deliver expressions
            output.expressions = expressions
            # Deliver extra targets
            for input in inputs:
                output.extra_targets = \
                    output.extra_targets.union(input.extra_targets)
            if extra_inputs is not None:
                for input in extra_inputs:
                    output.extra_targets.add(input.name)

        # 5. Refine the shape and data type
        outputs = _helper.OperatorHelper.apply(op_type,
            arguments=kwargs, inputs=inputs, outputs=outputs)

        # 6. Returns
        if num_outputs > 1: return outputs
        elif num_outputs == 1: return outputs[0]
        else: return None

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
        filler = _proto_def.TensorFillerProto()
        filler.tensor = self.name
        filler.type = type.lower()

        if filler.type in ['placeholder', 'variable']:
            pass
        elif filler.type == 'constant':
            filler.value = kwargs['value'] if 'value' in kwargs else 0
        elif filler.type in ['normal', 'gaussian']:
            filler.mean = kwargs['mean'] if 'mean' in kwargs else 0
            filler.std = kwargs['std'] if 'std' in kwargs else 1
            filler.type = 'normal'
        elif filler.type == 'uniform':
            filler.low = kwargs['low'] if 'low' in kwargs else 0
            filler.high = kwargs['high'] if 'high' in kwargs else 1
            filler.type = 'uniform'
        elif filler.type in ['truncated_normal', 'truncatednormal']:
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
        elif filler.type in ['glorot_uniform', 'xavier']:
            filler.scale = kwargs['scale'] if 'scale' in kwargs else 3.0
        elif filler.type in ['glorot_normal', 'msra']:
            filler.scale = kwargs['scale'] if 'scale' in kwargs else 2.0
        else:
            raise ValueError('Unknown filler type: {}'.format(filler.type))

        _workspace.CreateFiller(filler)
        return self