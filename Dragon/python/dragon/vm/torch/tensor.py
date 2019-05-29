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
import numpy

from dragon import config as _cfg
from dragon.core import mapping as _mapping
from dragon.core.tensor import Tensor as _Tensor
from dragon.core import proto_utils as _proto_utils
from dragon.core import tensor_utils as _tensor_utils
from dragon import get_default_workspace as _workspace

from dragon.vm.torch.c_api import Size as _Size
from dragon.vm.torch.c_api import device as _Device
from dragon.vm.torch.c_api import _get_tensor_pool
from dragon.vm.torch.c_api import from_dragon as _from_dragon


class Tensor(object):
    def __init__(self, *args, **kwargs):
        # Internal properties
        self._device = kwargs.get('device', _Device())
        self._requires_grad = kwargs.get('requires_grad', False)
        self._tensor = kwargs.get('name', None)
        self._own_storage = kwargs.get('own_storage', True)

        # Hold it to lock shared objects(i.e., tensor with same storage)
        self._ref_objects = []
        # Owned by the leaf variables(i.e. Can not be Reshaped)
        self._static_shape = None
        # Owned by the grad required variables
        self.__jit_recorder__ = self._ignored_grads = None
        # Whether this tensor should accumulate the gradients
        self.__accumulating__ = False

        # Constructor
        if len(args) == 0:
            # + empty tensor, not leaf
            if self._tensor is not None:
                _workspace().CreateTensor(self._tensor)
        elif len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                # + torch.Tensor(sequence)
                dtype = kwargs.get('dtype', 'float32')
                self._from_numpy(numpy.array(args[0], dtype=dtype))
            elif isinstance(args[0], numpy.ndarray):
                # + torch.Tensor(array)
                self._from_numpy(args[0], copy=kwargs.get('copy', True))
            else:
                # + class torch.Tensor(size)
                if not isinstance(args[0], six.integer_types):
                    raise ValueError('Excepted integer as size.')
                self._from_shape(args[0], kwargs.get('dtype', 'float32'))
        else:
            # + torch.Tensor(*sizes)
            if not all(isinstance(arg, six.integer_types) for arg in args):
                raise ValueError('Excepted integer(s) as sizes.')
            self._from_shape(args, kwargs.get('dtype', 'float32'))

        # Store the reference of backend
        self._storage = _workspace().GetTensor(
            self.name) if self.name is not None else None

    def _from_numpy(self, array, copy=True):
        self._static_shape = _Size(array.shape)
        # We use the scope of ``numpy`` instead of ``leaf``
        # As it is costly to switch memory between ``copy`` and ``zero-copy``
        if copy: array = array.copy()
        self._tensor = _tensor_utils.FromArray(
            array, _get_tensor_pool().get('${NUMPY}'))
        self._ignored_grads = {self.name + '_grad'} \
            if not self._requires_grad else None

    def _from_shape(self, shape, dtype):
        if isinstance(shape, six.integer_types): shape = [shape]
        self._static_shape = _Size(shape)
        self._tensor = _tensor_utils.FromShape(
            shape, dtype, _get_tensor_pool().get('${LEAF}'))
        self._ignored_grads = {self.name + '_grad'} \
            if not self._requires_grad else None

    @property
    def name(self):
        """Extent ``name`` for the graph-based representation.

        Returns
        -------
        str
            The name of dragon tensor.

        """
        return self._tensor.name if hasattr(
            self._tensor, 'name') else self._tensor

    @property
    def device(self):
        """Return the device of this tensor.

        Returns
        -------
        dragon.vm.torch.device
           The device.

        """
        return self._device

    def cpu(self):
        """Switch the internal storage on cpu memory.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        self._device.type = 'cpu'
        self._storage.ToCPU()
        return self

    def cuda(self, device=None):
        """Switch the internal storage on cuda memory.

        Parameters
        ----------
        device : int, optional
            The device index.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        if device is None: device = _cfg.GetGPU()
        self._storage.ToCUDA(device)
        self._device.type, self._device.index = 'cuda', device
        return self

    def numpy(self, readonly=False):
        """Create a numpy nd-array sharing this tensor.

        Parameters
        ----------
        readonly : boolean, optional, default=False
            Whether to sync the contents with device.

        Returns
        -------
        numpy.ndarray
            The numpy array.

        """
        return _tensor_utils.ToArray(self._tensor, readonly)

    def dragon(self):
        """Create a dragon tensor sharing this tensor.

        Returns
        -------
        dragon.Tensor
            The dragon tensor.

        """
        if isinstance(self._tensor, str):
            return _Tensor.Ref(self._tensor,
                shape=self.shape, dtype=self.dtype)
        else: return self._tensor

    def __add__(self, other):
        """Calculate x + y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.add(other)

    def __iadd__(self, other):
        """Calculate x = x + y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.add_(other)

    def __sub__(self, other):
        """Calculate x - y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.sub(other)

    def __isub__(self, other):
        """Calculate x = x - y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.sub_(other)

    def __mul__(self, other):
        """Calculate x * y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.mul(other)

    def __imul__(self, other):
        """Calculate x = x * y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.mul_(other)

    def __div__(self, other):
        """Calculate x / y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.div(other)

    def __idiv__(self, other):
        """Calculate x = x / y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.div_(other)

    def __truediv__(self, other):
        """Calculate x / y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.div(other)

    def __itruediv__(self, other):
        """Calculate x = x / y.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, int or float.
            The y.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.div_(other)

    def __neg__(self):
        """Calculate -x.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.mul(-1.0)

    def __gt__(self, other):
        """Compute *self* > *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor or number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        return self.gt(other)

    def __ge__(self, other):
        """Compute *self* >= *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor or number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        return self.ge(other)

    def __lt__(self, other):
        """Compute *self* < *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor or number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        return self.lt(other)

    def __le__(self, other):
        """Compute *self* <= *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor or number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        return self.le(other)

    def __eq__(self, other):
        """Compute *self* == *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor or number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        return self.eq(other)

    def __repr__(self):
        """Return a format str representing the internal storage.

        Returns
        -------
        str
            The format str.

        """
        np_data = self.numpy(readonly=True)
        if len(np_data.shape) == 0: return str(np_data)
        format_str = str(np_data)
        format_shape = 'x'.join([str(dim) for dim in np_data.shape])
        meta_info = '\n[torch.{} of size {}]'.\
            format(self._type2str(), format_shape)
        if self.device.type == 'cuda':
            meta_info = '\n[torch.cuda.{} of size {} (GPU {})]'.format(
                self._type2str(), format_shape, self.device.index)
        del np_data # DECREF
        return format_str + meta_info

    def __float__(self):
        """Return a float Python scalar of size-1 tensor.

        Returns
        -------
        float
            The float value.

        """
        if self.numel() == 1: return float(self.numpy(readonly=True))
        raise TypeError('Only size-1 array can be converted to Python scalars.')

    def __int__(self):
        """Return a int Python scalar of size-1 tensor.

        Returns
        -------
        int
            The int value.

        """
        return int(self.__float__())

    def __del__(self):
        if not self._requires_grad or self._static_shape:
            if self._own_storage and self._tensor:
                # Always reuse the leaf variables or
                # tensors that do not require grad
                # PyGC will detect them automatically
                _get_tensor_pool().put(self.name)

    def _process_indices(self, item):
        if not isinstance(item, (slice, tuple)):
            # + value[?]
            if not isinstance(item, int):
                raise ValueError('The index should be a integer.')
            item = (item,)
        if not isinstance(item, tuple):
            # + value[?:?]
            item = tuple([item])
        # + value[?:?, ?:?, ...]
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
        item : int, slice or dragon.vm.torch.Tensor
            The indices.

        Returns
        -------
        Tensor
            The output tensor.

        """
        if isinstance(item, Tensor):
            return self.masked_select(item)
        else:
            starts, sizes = self._process_indices(item)
            return self._index(starts, sizes)

    def __setitem__(self, key, value):
        """Set the value at the specific indices.

        Parameters
        ----------
        key : int, slice or dragon.vm.torch.Tensor
            The indices.
        value : number, sequence or dragon.vm.torch.Tensor
            The value.

        Returns
        -------
        None

        """
        if isinstance(key, Tensor):
            return self.masked_fill_(key, value)
        else:
            starts, sizes = self._process_indices(key)
            return self._assign(starts, sizes, value)

    def __hash__(self):
        return id(self)

    ##############################################
    #                                            #
    #                  PROPERTIES                #
    #                                            #
    ##############################################

    def size(self, axis=None):
        """Return the size of this tensor.

        Parameters
        ----------
        axis : int, optional
            The optional axis.

        Returns
        -------
        number or dragon.vm.torch.Size
            The size.

        """
        s = _Size(self._storage.dims)
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
        return self._storage.ndim

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
        return self._storage.size

    @property
    def dtype(self):
        """Return the data type of this tensor.

        Returns
        -------
        str
            The data type.

        """
        return self._storage.dtype

    def type(self, dtype=None):
        """Return the data type of this tensor.

        If ``dtype`` is not ``None``, cast ``self`` to the new tensor.

        Parameters
        ----------
        dtype : str, optional
            The specified type.

        Returns
        -------
        str or dragon.vm.torch.Tensor
            The data type or the new tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.type')

    def is_floating_point(self):
        """Whether the data type is floating.

        Floating types contains: *float16*, *float32*, *float64*

        Returns
        -------
        boolean
            *True* if the data type is floating.

        """
        return 'float' in self.dtype

    ##############################################
    #                                            #
    #                  OPERATORS                 #
    #                                            #
    ##############################################

    def squeeze(self, dim=None):
        """Return a tensor with all the dimensions of input of size 1 removed.

        Parameters
        ----------
        dim : int, optional
            The optional dim to remove.


        Returns
        -------
        dragon.vm.torch.Tensor
            The new tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.squeeze')

    def squeeze_(self, dim=None):
        """Inplace of ``Tensor.squeeze()``

        Parameters
        ----------
        dim : int, optional
            The optional dim to remove.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.squeeze_')

    def unsqueeze(self, dim):
        """Return a tensor with a dimension of size 1 inserted at the specified position.

        Parameters
        ----------
        dim : int
            The dim to insert.

        Returns
        -------
        dragon.vm.torch.Tensor
            The new tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.unsqueeze')

    def unsqueeze_(self, dim):
        """In-place version of ``Tensor.unsqueeze()``

        Parameters
        ----------
        dim : int
            The dim to insert.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.unsqueeze_')

    def view(self, *shape):
        """Return a new tensor with the same data but a different size.

        Parameters
        ----------
        shape : int...
            The new size.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.view')

    def reshape(self, *shape):
        """Return a new tensor with the same data but a different size.

        See also: *torch.view(*shape)*

        Parameters
        ----------
        shape : int...
            The new size.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.view(*shape)

    def view_as(self, other):
        """Return a new tensor with the same data but a different size as the given tensor.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor
            The tensor to guide the new size.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.view_as')

    def permute(self, *dims):
        """Return a new tensor with the specific order of dimensions.

        Parameters
        ----------
        dims : int...
            The new order of dimensions.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.permute')

    def narrow(self, dimension, start, length):
        """Return a new tensor that is a narrowed version of input tensor.

        Parameters
        ----------
        dimension : int
            The dimension to narrow.
        start : int
            The starting position.
        length : int
            The distance to the ending position.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.narrow')

    def repeat(self, *sizes):
        """Repeat elements along the specified dimensions.

        Parameters
        ----------
        sizes : int...
            The number of times to repeat.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.repeat')

    def chunk(self, chunks, dim=0):
        """Split self into several parts along the given axis.

        Parameters
        ----------
        chunks : int
            The number of chunks to split.
        dim : int, optional
            The dim to split.

        Returns
        -------
        sequence of dragon.vm.torch.Tensor
            The output chunks.

        """
        raise NotImplementedError('Refer torch.ops.tensor.chunk')

    def nonzero(self):
        """Return the indices of non-zero elements.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.nonzero')

    def where(self, condition, y):
        """Select elements from either ``self`` or ``y``, depending on ``condition``.

        Parameters
        ----------
        condition : dragon.vm.torch.Tensor
            The byte condition tensor.
        y : dragon.vm.torch.Tensor
            The elements for *0*.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.where')

    def index_select(self, dim, index):
        """Select the values along the given axis using index.

        Parameters
        ----------
        dim : int
            The dim to gather.
        index : dragon.vm.torch.Tensor
            The indices.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.index_select')

    def masked_select(self, mask):
        """Select the values where mask is *1*.

        Parameters
        ----------
        mask : dragon.vm.torch.Tensor
            The mask to select values.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.masked_select')

    def copy_(self, src, non_blocking=False):
        """Copy the elements from ``src`` into this tensor and return ``self``.

        Parameters
        ----------
        src : dragon.vm.torch.Tensor
            The source tensor.
        non_blocking : boolean, optional, default=False
            Whether to copy asynchronously between CPU and GPU.

        Returns
        -------
        dragon.vm.torch.Tensor
            The ``self`` tensor.

        """
        # Copy memory
        _tensor_utils.FromTensor(
            src, _proto_utils.GetDeviceOption(
                src.device.type, src.device.index),
            self.name, _proto_utils.GetDeviceOption(
                self.device.type, self.device.index))
        # Transfer the static shape if necessary
        self._static_shape = src.size() \
            if self._static_shape else None
        return self

    def fill_(self, value):
        """Fill self with the given value.

        Parameters
        ----------
        value : number
            The value to fill.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.fill_')

    def masked_fill_(self, mask, value):
        """Fill self with the given value where ``mask`` is *1*.

        Parameters
        ----------
        mask : dragon.vm.torch.Tensor
            The mask.
        value : number
            The value to fill.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.masked_fill_')

    def zero_(self):
        """Fill self tensor with zeros.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        self.fill_(0.)

    def one_(self):
        """Fill self tensor with ones.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        self.fill_(1.)

    def uniform_(self, low=0, high=1):
        """Fill self tensor with the specified uniform distribution.

        Parameters
        ----------
        low : number, optional, default=0
            The lower bound.
        high : number, optional, default=1
            The higher bound.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.uniform_')

    def normal_(self, mean=0, std=1):
        """Fill self with the specified normal distribution.

        Parameters
        ----------
        mean : number, optional, default=0
            The mean(mu) of normal distribution.
        std : number, optional, default=1
            The std(sigma) of normal distribution.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.normal_')

    def multinomial(self, num_samples, eps=0.):
        """Return a tensor where each row contains ``num_samples``,
           sampled from the multinomial distribution.

        Parameters
        ----------
        num_samples : int
            The number of samples.
        eps : float, optional, default=0.
            The prob to a uniform sampling.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.multinomial')

    def add(self, value):
        """Add the ``self`` and ``value`` into the output tensor.

        Parameters
        ----------
        value : dragon.vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.add')

    def add_(self, value):
        """In-place version of ``Tensor.add()``.

        Parameters
        ----------
        value : dragon.vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.add_')

    def sub(self, value):
        """Subtract the ``self`` and ``value`` into the output tensor.

        Parameters
        ----------
        value : dragon.vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.sub')

    def sub_(self, value):
        """In-place version of ``Tensor.sub()``

        Parameters
        ----------
        value : dragon.vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.sub_')

    def mul(self, value):
        """Multiply the ``self`` and ``value`` into the output tensor.

        Parameters
        ----------
        value : dragon.vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.mul')

    def mul_(self, value):
        """In-place version of ``Tensor.mul()``

        Parameters
        ----------
        value : dragon.vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.mul_')

    def div(self, value):
        """Divide the ``self`` and ``value`` into the output tensor.

        Parameters
        ----------
        value : dragon.vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.div')

    def div_(self, value):
        """In-place version of ``Tensor.div()``

        Parameters
        ----------
        value : dragon.vm.torch.Tensor, int or float
            The value tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        raise NotImplementedError('Refer torch.ops.tensor.div_')

    def clamp(self, min=None, max=None):
        """Return a tensor that all elements are clamped into the range [min, max].

        Parameters
        ----------
        min : number, optional
            The min value.
        max : number, optional
            The max value.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.clamp')

    def clamp_(self, min=None, max=None):
        """Clamp all elements are clamped into the range [min, max].

        Parameters
        ----------
        min : number, optional
            The min value.
        max : number, optional
            The max value.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.clamp_')

    def log(self):
        """Compute the natural logarithm of this tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The log tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.log')

    def exp(self):
        """Compute the exponential of this tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The exp tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.exp')

    def sqrt(self):
        """Compute the square-root of this tensor.

        Returns
        -------
        torch.Tensor
            The sqrt tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.sqrt')

    def mean(self, dim=None, keepdim=False):
        """Returns the mean of all elements or elements along the given dim.

        Parameters
        ----------
        dim : int, optional
            The axis of tensor to compute mean value.
        keepdim : bool, optional, default=False
            Whether the output tensor has dim retained or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The mean-reduced tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.mean')

    def sum(self, dim=None, keepdim=False):
        """Returns the sum of all elements or elements along the given dim.

        Parameters
        ----------
        dim : int, optional
            The axis of tensor to compute sum value.
        keepdim : bool, optional, default=False
            Whether the output tensor has dim retained or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The sum-reduced tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.sum')

    def max(self, dim=None, keepdim=False):
        """Return the values and indices of maximum elements along the given axis.

        Parameters
        ----------
        dim : int, optional
            The axis of tensor to compute sum value.
        keepdim : bool, optional, default=False
            Whether the output tensor has dim retained or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The maximum values and indices.

        """
        raise NotImplementedError('Refer torch.ops.tensor.max')

    def min(self, dim=None, keepdim=False):
        """Return the values and indices of minimum elements along the given axis.

        Parameters
        ----------
        dim : int, optional
            The axis of tensor to compute sum value.
        keepdim : bool, optional, default=False
            Whether the output tensor has dim retained or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The minimum values and indices.

        """
        raise NotImplementedError('Refer torch.ops.tensor.min')

    def gt(self, other):
        """Compute *self* > *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.gt')

    def ge(self, other):
        """Compute *self* >= *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.ge')

    def lt(self, other):
        """Compute *self* < *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.lt')

    def le(self, other):
        """Compute *self* <= *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.le')

    def eq(self, other):
        """Compute *self* == *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.eq')

    def ne(self, other):
        """Compute *self* != *other* element-wise.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor, number
            The other tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.ne')

    def half(self):
        """Return a *float16* tensor with elements of ``self``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The half tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.half')

    def half_(self):
        """In-place version of ``Tensor.half()``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The half tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.half_')

    def float(self):
        """Return a ``float32`` tensor with elements of ``self``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The float tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.float')

    def float_(self):
        """In-place version of ``Tensor.float()``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The float tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.float_')

    def double(self):
        """Return a *float64* tensor with elements of ``self``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The double tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.double')

    def double_(self):
        """In-place version of ``Tensor.double()``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The double tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.double_')

    def int(self):
        """Return a *int32* tensor with elements of ``self``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The int tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.int')

    def int_(self):
        """In-place version of ``Tensor.int()``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The int tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.int_')

    def long(self):
        """Return a *int64* tensor with elements of ``self``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The long tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.long')

    def long_(self):
        """In-place version of ``Tensor.long()``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The long tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.long_')

    def byte(self):
        """Return a *uint8* tensor with elements of ``self``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.byte')

    def byte_(self):
        """In-place version of ``Tensor.byte()``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.byte_')

    def char(self):
        """Return a *int8* tensor with elements of ``self``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.char')

    def char_(self):
        """In-place version of ``Tensor.char()``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The byte tensor.

        """
        raise NotImplementedError('Refer torch.ops.tensor.char_')

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
        return Tensor(device=self.device, name=self.name, own_storage=False)

    def detach(self):
        return self.data

    @property
    def grad(self):
        g = _from_dragon(self.name + '_grad', False)
        if g: g._static_shape = self.shape
        return g

    @property
    def grad_fn(self):
        return True if self.__jit_recorder__ \
            and len(self.__jit_recorder__.ops) > 0 else None

    def backward(self, gradient=None):
        raise NotImplementedError('Refer torch.autograd.variable.backward().')

    ##############################################
    #                                            #
    #                   DEBUG                    #
    #                                            #
    ##############################################

    def DebugExpression(self):
        return self.__jit_recorder__.debug_str(self.name)

    ##############################################
    #                                            #
    #                   MISC                     #
    #                                            #
    ##############################################

    def _type2str(self):
        return _mapping.TENSOR_TYPE_TO_TORCH_TENSOR[self.dtype]


def CharTensor(*args, **kwargs):
    kwargs['dtype'] = 'int8'
    return Tensor(*args, **kwargs)


def ByteTensor(*args, **kwargs):
    kwargs['dtype'] = 'uint8'
    return Tensor(*args, **kwargs)


def IntTensor(*args, **kwargs):
    kwargs['dtype'] = 'int32'
    return Tensor(*args, **kwargs)


def LongTensor(*args, **kwargs):
    kwargs['dtype'] = 'int64'
    return Tensor(*args, **kwargs)


def HalfTensor(*args, **kwargs):
    kwargs['dtype'] = 'float16'
    return Tensor(*args, **kwargs)


def FloatTensor(*args, **kwargs):
    kwargs['dtype'] = 'float32'
    return Tensor(*args, **kwargs)


def DoubleTensor(*args, **kwargs):
    kwargs['dtype'] = 'float64'
    return Tensor(*args, **kwargs)


def _LeafTensor(shape, dtype='float32', device=_Device(), requires_grad=False):
    """Create a torch tensor according to shape, dtype and device.

    Commonly used to create leaf variables, i.e., the parameters or placeholders.

    """
    constructor = globals()[_mapping.TENSOR_TYPE_TO_TORCH_TENSOR[dtype]]
    return constructor(*shape, device=device, requires_grad=requires_grad)


def _RuntimeTensor(name, dtype='float32', device=_Device()):
    """Create a torch tensor according to dtype and device.

    Commonly used to represent the outputs that are hard to compute shape,
    i.e., the shape is computed by the backend automatically.

    """
    constructor = globals()[_mapping.TENSOR_TYPE_TO_TORCH_TENSOR[dtype]]
    return constructor(name=name, device=device)


def _ReferenceTensor(src):
    """Create a reference from source tensor.

    Commonly used to hold the same storage but takes different sizes,
    i.e., view, squeeze, and unsqueeze.

    """
    constructor = globals()[_mapping.TENSOR_TYPE_TO_TORCH_TENSOR[src.dtype]]
    T = constructor(name=_get_tensor_pool().get('${REFERENCE}'), device=src.device)
    T._ref_objects.append(src)
    return T


class Parameter(Tensor):
    def __init__(self, tensor, requires_grad=True):
        super(Parameter, self).__init__(name=tensor.name)
        self.__dict__ = tensor.__dict__
        self.requires_grad = requires_grad
        self._th_tensor = tensor

    def __repr__(self):
        return 'Parameter containing:\n' + \
            super(Parameter, self).__repr__()


def tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """Construct a tensor with data.

    Parameters
    ----------
    data : array_like
        The data to initialize.
    dtype : str, optional
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device option.
    requires_grad : boolean, optional, default=False
        Whether to enable auto-grad.
    pin_memory : boolean, optional, default=False
        Whether to allocate pin-memory for cpu tensor.

    """
    data = numpy.array(data)
    if dtype is None: dtype = str(data.dtype)
    else: data = data.astype(dtype)
    if device is None: device = _Device()
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)