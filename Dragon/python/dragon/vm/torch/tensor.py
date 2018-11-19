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

import six
import numpy as np
import dragon as dg
import dragon.core.tensor_utils as tensor_utils

from dragon.vm.torch.tensor_uitls import from_dragon
from dragon.vm.torch.tensor_pool import TPool
from dragon.vm.torch.constants import CTX_TO_DEVICE_OPTION
from .c_apis import *


__all__ = [
    'Tensor', 'Parameter',
    'FloatTensor', 'DoubleTensor',
    'IntTensor', 'LongTensor',
    'ByteTensor', 'CharTensor',
]


##############################################
#                                            #
#                 Tensor-Core                #
#                                            #
##############################################


class Tensor(object):
    def __init__(self, *args, **kwargs):
        # Optional arguments
        name = kwargs.get('name', None)

        # Internal properties
        self._ctx = kwargs.get('ctx', None)
        self._ctx = list(self._ctx) if self._ctx is not None else ['CPU', 0]
        self._dtype = kwargs.get('dtype', 'float32')
        self._requires_grad = kwargs.get('requires_grad', False)
        self._dg_tensor = kwargs.get('dg_tensor', None)
        self._own_storage = kwargs.get('own_storage', True)

        # Hold it to lock shared objects(i.e., tensor with same storage)
        self._ref_objects = []
        # Owned by the leaf variables(i.e. Can not be Reshaped)
        self._static_shape = None
        # Owned by the grad required variables
        self._expr = self._ignored_grads = None

        # Constructor
        if len(args) == 0:
            # + empty tensor, not leaf
            if name is not None: self._dg_tensor = name
        elif len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                # + torch.Tensor(sequence)
                self._init_from_numpy(np.array(args[0], dtype=self._dtype))
            elif isinstance(args[0], np.ndarray):
                # + torch.Tensor(nd-array)
                self._init_from_numpy(args[0])
            else:
                # + class torch.Tensor(size)
                if not isinstance(args[0], six.integer_types):
                    raise ValueError('Excepted integer as size.')
                self._init_from_shape(args[0])
        else:
            # + torch.Tensor(*sizes)
            if not all(isinstance(arg, six.integer_types) for arg in args):
                raise ValueError('Excepted integer(s) as sizes.')
            self._init_from_shape(shape=args)

    def _init_from_numpy(self, array):
        self._static_shape = Size(array.shape)
        # We use the scope of ``numpy`` instead of ``leaf``
        # As it is costly to switch memory between ``copy`` and ``zero-copy``
        self._dg_tensor = tensor_utils.FromPyArray(array, name=TPool.get('numpy'))
        self._ignored_grads = {self.name + '_grad'} if not self._requires_grad else None

    def _init_from_shape(self, shape):
        if isinstance(shape, six.integer_types): shape = [shape]
        self._static_shape = Size(shape)
        self._dg_tensor = tensor_utils.FromShape(shape, self._dtype,
                ctx=CTX_TO_DEVICE_OPTION[tuple(self._ctx)], name=TPool.get('leaf'))
        self._ignored_grads = {self.name + '_grad'} if not self._requires_grad else None

    @property
    def name(self):
        """Extent ``name`` for the graph-based representation.

        Returns
        -------
        str
            The name of dragon tensor.

        """
        if isinstance(self._dg_tensor, str): return self._dg_tensor
        return self._dg_tensor.name

    def cpu(self):
        """Switch the internal storage on cpu memory.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        self._ctx[0] = 'CPU'
        tensor_utils.ToCPUTensor(self._dg_tensor)
        return self

    def cuda(self, device=None):
        """Switch the internal storage on cuda memory.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        if device is None: device = dg.config.GetGPU()
        tensor_utils.ToCUDATensor(self._dg_tensor, device)
        self._ctx[0], self._ctx[1] = ('CUDA', device)
        return self

    def numpy(self):
        """Create a numpy nd-array sharing this tensor.

        Returns
        -------
        numpy.ndarray
            The numpy nd-array.

        """
        if self._ctx[0] != 'CPU':
            raise RuntimeError("can't convert CUDA tensor to numpy "
                               "(it doesn't support GPU arrays). "
                               "Use .cpu() to move the tensor to host memory first.")
        return tensor_utils.ToPyArray(self._dg_tensor)

    def numpy_ex(self):
        """Create a numpy const nd-array sharing this tensor.

        Returns
        -------
        numpy.ndarray
            The numpy nd-array.

        """
        return tensor_utils.ToPyArrayEx(self._dg_tensor)

    def dragon(self):
        """Create a dragon tensor sharing this tensor.

        Returns
        -------
        Tensor
            The dragon tensor.

        """
        if isinstance(self._dg_tensor, str):
            return dg.Tensor(_name=self._dg_tensor)
        else: return self._dg_tensor

    def __add__(self, other):
        """Calculate x + y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float
            The y.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        return self.add(other)

    def __iadd__(self, other):
        """Calculate x = x + y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float
            The y.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        return self.add_(other)

    def __sub__(self, other):
        """Calculate x - y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        return self.sub(other)

    def __isub__(self, other):
        """Calculate x = x - y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float
            The y.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        return self.sub_(other)

    def __mul__(self, other):
        """Calculate x * y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        return self.mul(other)

    def __imul__(self, other):
        """Calculate x = x * y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        return self.mul_(other)

    def __div__(self, other):
        """Calculate x / y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        return self.div(other)

    def __idiv__(self, other):
        """Calculate x = x / y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        return self.div_(other)

    def __truediv__(self, other):
        """Calculate x / y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        return self.div(other)

    def __itruediv__(self, other):
        """Calculate x = x / y.

        Parameters
        ----------
        other : vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        return self.div_(other)

    def __neg__(self):
        """Calculate -x.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        return self.mul(-1.0)

    def __repr__(self):
        """Return a format str representing the internal storage.

        Returns
        -------
        str
            The format str.

        """
        np_data = tensor_utils.ToPyArrayEx(self._dg_tensor) # const reference
        if len(np_data.shape) == 0: return str(np_data)
        format_str = str(np_data)
        meta_info = '\n[torch.{} of size {}]'.\
            format(self._type2str(), 'x'.join([str(dim) for dim in np_data.shape]))
        if self._ctx[0] == 'CUDA':
            meta_info = '\n[torch.cuda.{} of size {} (GPU {})]'. \
                format(self._type2str(), 'x'.join([str(dim) for dim in np_data.shape]), self._ctx[1])
        del np_data  # DECREF
        return format_str + meta_info

    def __del__(self):
        if not self._requires_grad or self._static_shape:
            if self._own_storage and self._dg_tensor:
                # Always reuse the leaf variables or
                # tensors that do not require grad
                # PyGC will detect them automatically
                TPool.put(self.name)

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
            # + value[?]
            if not isinstance(item, int):
                raise ValueError('The index should be a integer.')
            item = (item,)
        if not isinstance(item, tuple):
            # + value[?:?]
            item = tuple([item])
        # + value[?:?, ?:?, ...]
        starts = []; ends = []
        input_shape = self.shape
        for ix, it in enumerate(item):
            keep_dims = True
            if isinstance(it, slice):
                # handle start
                if it.start is None: starts.append(0)
                else: starts.append(it.start)
                # handle stop
                if it.stop is None: ends.append(input_shape[ix])
                else: ends.append(it.stop)
                # handle step
                if it.step is not None:
                    raise NotImplementedError('Indexing with step has not been implemented yet. ')
            elif isinstance(it, int):
                starts.append(it)
                ends.append(it + 1)
                keep_dims = False
            else:
                raise TypeError('Unsupported type of indices: {}'.format(type(type(it))))
            # check boundaries
            if starts[-1] < 0 or starts[-1] >= input_shape[ix]:
                raise ValueError('The cropping starts at the pos {} of axis {},'
                                '\nwhile the dimension of this axis is {}.'
                                 .format(starts[-1], ix, input_shape[ix]))
            if ends[-1] < 0 or ends[-1] > input_shape[ix]:
                raise ValueError('The cropping ends at the pos {} of axis {},'
                                '\nwhile the dimension of this axis is {}.'
                                 .format(ends[-1], ix, input_shape[ix]))
            if starts[-1] == ends[-1]:
                raise ValueError('The cropping starts and ends of axis {} '
                                 'can not be equal, got {}:{}.'
                                 .format(ix, starts[-1], ends[-1]))
            # fake ends for squeezing dimensions
            # TODO(PhyscalX): Can we have a better solution to be compatible with the backend?
            if not keep_dims: ends[-1] = -1
        return self._crop(starts, ends)

    def __setitem__(self, key, value):
        print(key, value.name)

    def device(self):
        return self._ctx[1]

    ##############################################
    #                                            #
    #                  PROPERTIES                #
    #                                            #
    ##############################################

    def size(self, axis=None):
        """Return the size of this tensor.

        Parameters
        ----------
        axis : int or None
            The optional axis.

        Returns
        -------
        vm.torch.Size or int
            The size.

        """
        # Improve the performance when frequently requiring the shape
        if self._static_shape: s = self._static_shape
        else: s = Size(tensor_utils.GetTensorInfo(self._dg_tensor, 2)['shape'])
        return s[axis] if axis is not None else s

    @property
    def shape(self):
        """Return the shape of this tensor.

        Returns
        -------
        vm.torch.Size
            The shape.

        """
        return self.size()

    def dim(self):
        """Return the number of dimensions of this tensor.

        Returns
        -------
        int
            The number of dimensions.

        """
        return len(self.size())

    def ndimension(self):
        """Alias for ``dim()``.

        Returns
        -------
        int
            The number of dimensions.

        """
        return self.dim()

    def numel(self):
        """Return the total number of elements in this tensor.

        Returns
        -------
        int
            The total count of elements.

        """
        return np.prod(self.shape)

    @property
    def dtype(self):
        """Return the data type of this tensor.

        Returns
        -------
        str
            The data type.

        """
        return self._dtype

    def type(self, dtype=None):
        """Return the data type of this tensor.

        If ``dtype`` is not ``None``, cast ``self`` to the new tensor.

        Parameters
        ----------
        dtype : str
            The specified type.

        Returns
        -------
        str or vm.torch.Tensor
            The data type or the new tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.type')


    ##############################################
    #                                            #
    #                  OPERATORS                 #
    #                                            #
    ##############################################

    def squeeze(self, dim=None):
        """Returns a tensor with all the dimensions of input of size 1 removed.

        Parameters
        ----------
        dim : int
            The optional dim to remove.


        Returns
        -------
        vm.torch.Tensor
            The new tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin._squeeze')

    def squeeze_(self, dim=None):
        """Inplace of ``Tensor.squeeze()``

        Parameters
        ----------
        dim : int
            The optional dim to remove.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.builtin._squeeze_')

    def unsqueeze(self, dim):
        """Returns a tensor with a dimension of size 1 inserted at the specified position.

        Parameters
        ----------
        dim : int
            The dim to insert.

        Returns
        -------
        vm.torch.Tensor
            The new tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin._unsqueeze')

    def unsqueeze_(self, dim):
        """Inplace of ``Tensor.unsqueeze()``

        Parameters
        ----------
        dim : int
            The dim to insert.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.builtin._unsqueeze_')

    def view(self, *args):
        """Return a new tensor with the same data but a different size.

        Parameters
        ----------
        args : tuple or int
            The new size.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.view')

    def view_as(self, other):
        """Return a new tensor with the same data but a different size as the given tensor.

        Parameters
        ----------
        other : vm.torch.Tensor
            The tensor to guide the new size.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.view_as')

    def permute(self, dims=None):
        """Return a new tensor with the specific order of dimensions.

        Parameters
        ----------
        dims : list of int
            The new order of dimensions.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.permute')

    def repeat(self, *sizes):
        """Repeat this tensor along the specified dimensions.

        Parameters
        ----------
        sizes : vm.torch.Size or int...
            The number of times to repeat.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.repeat')

    def copy_(self, src, non_blocking=False):
        """Copy the elements from ``src`` into this tensor and return ``self``.

        Parameters
        ----------
        src : vm.torch.Tensor
            The source tensor.
        non_blocking : boolean
            Whether to copy asynchronously between CPU and GPU.

        Returns
        -------
        vm.torch.Tensor
            The ``self`` tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.copy_')

    def fill_(self, value):
        """Fills self tensor with the specified value.

        Parameters
        ----------
        value : numerical type

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.builtin.fill_')

    def zero_(self):
        """Fills self tensor with zeros.

        Parameters
        ----------
        value : numerical type

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        self.fill_(0.)

    def one_(self):
        """Fills self tensor with ones.

        Parameters
        ----------
        value : numerical type

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        self.fill_(1.)

    def uniform_(self, low=0, high=1):
        """Fill self tensor with the specified uniform distribution.

        Parameters
        ----------
        low : numerical type
            The lower bound.
        high : numerical type
            The higher bound.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.builtin.uniform_')

    def normal_(self, mean=0, std=1):
        """Fill self tensor with the specified normal distribution.

        Parameters
        ----------
        mean : numerical type
            The mean(mu) of normal distribution.
        std : numerical type
            The std(sigma) of normal distribution.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.builtin.normal_')

    def add(self, value):
        """See ``torch.add()``

        Parameters
        ----------
        value : vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.add')

    def add_(self, value):
        """Inplace of ``torch.add()``

        Parameters
        ----------
        value : vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.builtin.add_')

    def sub(self, value):
        """Subtract the ``self`` and ``value`` into the output tensor.

        Parameters
        ----------
        value : vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.sub')

    def sub_(self, value):
        """Inplace of ``Tensor.sub()``

        Parameters
        ----------
        value : vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.builtin.sub_')

    def mul(self, value):
        """Multiply the ``self`` and ``value`` into the output tensor.

        Parameters
        ----------
        value : vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.mul')

    def mul_(self, value):
        """Inplace of ``Tensor.mul()``

        Parameters
        ----------
        value : vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.builtin.mul_')

    def div(self, value):
        """Divide the ``self`` and ``value`` into the output tensor.

        Parameters
        ----------
        value : vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.div')

    def div_(self, value):
        """Inplace of ``Tensor.div()``

        Parameters
        ----------
        value : vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.builtin.div_')

    def clamp(self, min=None, max=None):
        """Return a tensor that all elements are clamped into the range [min, max].

        Parameters
        ----------
        min : numerical or None
            The min value.
        max : numerical or None
            The max value.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.clamp')

    def clamp_(self, min=None, max=None):
        """Clamp all elements are clamped into the range [min, max].

        Parameters
        ----------
        min : numerical or None
            The min value.
        max : numerical or None
            The max value.

        Returns
        -------
        vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.clamp_')

    def log(self):
        """Compute the natural logarithm of this tensor.

        Parameters
        ----------
        None

        Returns
        -------
        vm.torch.Tensor
            The log tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.log')

    def exp(self):
        """Compute the exponential of this tensor.

        Parameters
        ----------
        None

        Returns
        -------
        vm.torch.Tensor
            The exp tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.exp')

    def mean(self, dim=None, keepdim=False):
        """Returns the mean of all elements or elements along the given dim.

        Parameters
        ----------
        dim : int or None
            The axis of tensor to compute mean value.
        keepdim : boolean
            Whether the output tensor has dim retained or not.

        Returns
        -------
        vm.torch.Tensor
            The mean-reduced tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.mean')

    def sum(self, dim=None, keepdim=False):
        """Returns the sum of all elements or elements along the given dim.

        Parameters
        ----------
        dim : int or None
            The axis of tensor to compute sum value.
        keepdim : boolean
            Whether the output tensor has dim retained or not.

        Returns
        -------
        vm.torch.Tensor
            The sum-reduced tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.sum')

    def max(self, dim=None, keepdim=False):
        """Return the values and indices of maximum elements along the given axis.

        Parameters
        ----------
        dim : int or None
            The axis of tensor to compute sum value.
        keepdim : boolean
            Whether the output tensor has dim retained or not.

        Returns
        -------
        vm.torch.Tensor
            The maximum values and indices.

        """
        raise NotImplementedError('Refer torch.ops.builtin.max')

    def min(input, dim=None, keepdim=False):
        """Return the values and indices of minimum elements along the given axis.

        Parameters
        ----------
        dim : int or None
            The axis of tensor to compute sum value.
        keepdim : boolean
            Whether the output tensor has dim retained or not.

        Returns
        -------
        vm.torch.Tensor
            The minimum values and indices.

        """
        raise NotImplementedError('Refer torch.ops.builtin.min')

    def half(self):
        """Return a ``float16`` tensor with elements of ``self``.

        Returns
        -------
        vm.torch.Tensor
            The half tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.half')

    def half_(self):
        """Inplace of ``Tensor.half()``.

        Returns
        -------
        vm.torch.Tensor
            The half tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.half_')

    def float(self):
        """Return a ``float32`` tensor with elements of ``self``.

        Returns
        -------
        vm.torch.Tensor
            The float tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.float')

    def float_(self):
        """Inplace of ``Tensor.float()``.

        Returns
        -------
        vm.torch.Tensor
            The float tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.float_')

    def double(self):
        """Return a ``float64`` tensor with elements of ``self``.

        Returns
        -------
        vm.torch.Tensor
            The double tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.double')

    def double_(self):
        """Inplace of ``Tensor.double()``.

        Returns
        -------
        vm.torch.Tensor
            The double tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.double_')

    def int(self):
        """Return a ``int32`` tensor with elements of ``self``.

        Returns
        -------
        vm.torch.Tensor
            The int tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.int')

    def int_(self):
        """Inplace of ``Tensor.int()``.

        Returns
        -------
        vm.torch.Tensor
            The int tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.int_')

    def long(self):
        """Return a ``int64`` tensor with elements of ``self``.

        Returns
        -------
        vm.torch.Tensor
            The long tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.long')

    def long_(self):
        """Inplace of ``Tensor.long()``.

        Returns
        -------
        vm.torch.Tensor
            The long tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.long_')

    def byte(self):
        """Return a ``uint8`` tensor with elements of ``self``.

        Returns
        -------
        vm.torch.Tensor
            The byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.byte')

    def byte_(self):
        """Inplace of ``Tensor.byte()``.

        Returns
        -------
        vm.torch.Tensor
            The byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.byte_')

    def char(self):
        """Return a ``int8`` tensor with elements of ``self``.

        Returns
        -------
        vm.torch.Tensor
            The byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.char')

    def char_(self):
        """Inplace of ``Tensor.char()``.

        Returns
        -------
        vm.torch.Tensor
            The byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.builtin.char_')

    ##############################################
    #                                            #
    #                  AUTO-GRAD                 #
    #                                            #
    ##############################################

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value
        self._ignored_grads = {self.name + '_grad'} if not value else None

    def volatile(self):
        raise NotImplementedError('Refer torch.autograd.variable.volatile().')

    @property
    def data(self):
        return Tensor(dtype=self.dtype, ctx=self._ctx,
            name=self.name, own_storage=False)

    @property
    def grad(self):
        g = from_dragon(self.name + '_grad', False)
        if g: g._static_shape = self.shape
        return g

    @property
    def grad_fn(self):
        return True if self._expr and len(self._expr._ops) > 0 else None

    def backward(self, gradient=None):
        raise NotImplementedError('Refer torch.autograd.variable.backward().')

    ##############################################
    #                                            #
    #                   DEBUG                    #
    #                                            #
    ##############################################

    def DebugExpression(self):
        return self._expr.debug_str(self.name)

    ##############################################
    #                                            #
    #                   MISC                     #
    #                                            #
    ##############################################

    def _type2str(self):
        return {
            'float16': 'HalfTensor',
            'float32': 'FloatTensor',
            'float64': 'DoubleTensor',
            'int32': 'IntTensor',
            'int64': 'LongTensor',
            'uint8': 'ByteTensor'
        }[self._dtype]


def HalfTensor(*args, **kwargs):
    kwargs['dtype'] = 'float16'
    return Tensor(*args, **kwargs)


def FloatTensor(*args, **kwargs):
    kwargs['dtype'] = 'float32'
    return Tensor(*args, **kwargs)


def DoubleTensor(*args, **kwargs):
    kwargs['dtype'] = 'float64'
    return Tensor(*args, **kwargs)


def IntTensor(*args, **kwargs):
    kwargs['dtype'] = 'int32'
    return Tensor(*args, **kwargs)


def LongTensor(*args, **kwargs):
    kwargs['dtype'] = 'int64'
    return Tensor(*args, **kwargs)


def ByteTensor(*args, **kwargs):
    kwargs['dtype'] = 'uint8'
    return Tensor(*args, **kwargs)


def CharTensor(*args, **kwargs):
    kwargs['dtype'] = 'int8'
    return Tensor(*args, **kwargs)


_DTYPE_TO_TENSOR = {
    'float16': HalfTensor,
    'float32': FloatTensor,
    'float64': DoubleTensor,
    'int32': IntTensor,
    'int64': LongTensor,
    'uint8': ByteTensor,
    'int8': CharTensor,
}


def LeafTensor(shape, dtype='float32', ctx=None, requires_grad=False):
    """Create a torch tensor according to shape, dtype and ctx.

    Commonly used to create leaf variables, i.e., the parameters or placeholders.

    """
    constructor = _DTYPE_TO_TENSOR[dtype]
    return constructor(*shape, ctx=ctx, requires_grad=requires_grad)


def RuntimeTensor(name, dtype='float32', ctx=None):
    """Create a torch tensor according to dtype and ctx.

    Commonly used to represent the outputs that are hard to compute shape,
    i.e., the shape is computed by the backend automatically.

    """
    constructor = _DTYPE_TO_TENSOR[dtype]
    dg.workspace.CreateTensor(name)
    return constructor(dg_tensor=name, ctx=ctx)


def ReferneceTensor(src):
    """Create a reference from source tensor.

    Commonly used to hold the same storage but takes different sizes,
    i.e., view, squeeze, and unsqueeze.

    """
    constructor = _DTYPE_TO_TENSOR[src._dtype]
    ref = constructor(dg_tensor=src.name, ctx=src._ctx)
    name = '{}/id:{}'.format(
        src.name.replace('[TPool]', '[Ref]'), id(ref))
    dg.workspace.CreateTensor(name)
    ref._dg_tensor, ref._own_storage = name, False
    ref._ref_objects.append(src)
    return ref


##############################################
#                                            #
#               Tensor-Extension             #
#                                            #
##############################################


class Parameter(Tensor):
    def __init__(self, tensor, requires_grad=True):
        super(Parameter, self).__init__()
        self.__dict__ = tensor.__dict__
        self.requires_grad = requires_grad
        self._th_tensor = tensor

    def __repr__(self):
        return 'Parameter containing:\n' + \
               super(Parameter, self).__repr__()
