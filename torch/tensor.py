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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.framework import config
from dragon.core.framework import context
from dragon.core.framework import mapping
from dragon.core.framework import proto_util
from dragon.core.framework import workspace
from dragon.core.util import math_util
from dragon.core.util import six
from dragon.vm.torch import cpp


class Tensor(object):
    """A multi-dimensional array containing elements of a single data type.

    To create a tensor from constant value, use ``torch.tensor(...)``:

    ```python
    # Create a constant tensor
    # The value is 1, dimensions is (0,)
    const_tensor = torch.tensor(1, dtype='float32')
    ```

    Besides, following initializers can also be used:

    ```python
    # Create an empty float32 tensor
    # The dimensions is (1, 2)
    empty_tensor = torch.empty(1, 2, dtype='float32')

    # Create a float32 tensor filling ``one`` or ``zero``
    ones = torch.ones(1, 2, dtype='float32')
    zeros = torch.zeros(1, 2, dtype='float32)
    ```

    Construct a tensor with device and grad will sometimes be helpful:

    ```python
    # Initialize a weight and bias on the gpu:0, whose gradient should not be ignored
    weight = torch.ones(1, 2, device=torch.device('cuda', 0), requires_grad=True)
    bias = torch.tensor(0, device=torch.device('cuda', 0), requires_grad=True)
    ```

    Be careful to store a tensor object, or the memory will not be free:

    ```python
    # The memory of ``my_tensor`` will be held
    # until the reference count decrease to zero
    my_object.tensors = []
    my_object.tensors.append(my_tensor)
    ```

    """
    def __init__(self, *args, **kwargs):
        # Internal properties
        self._id = kwargs.get('id', None)
        self._device = kwargs.get('device', cpp.device())
        self._requires_grad = kwargs.get('requires_grad', False)
        self._own_storage = kwargs.get('own_storage', True)
        self._const_size = None  # Attribute to represent a leaf variable
        self._ignored_grads = set()  # Blacklist of the non-gradient variables
        self.__tape__ = None  # Instance tape to record operations
        self.__accumulating__ = False  # Flag for gradient accumulating

        # Constructor
        if len(args) == 0:
            # >>> Empty tensor
            if self._id is not None:
                ws = workspace.get_workspace()
                self.__gc__ = ws.collectors.TENSOR
                self._impl = ws.CreateTensor(self._id)
            else:
                self.__gc__ = None
        elif len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                # >>> torch.Tensor(sequence)
                dtype = kwargs.get('dtype', 'float32')
                self._from_numpy(numpy.array(args[0], dtype=dtype), copy=False)
            elif isinstance(args[0], numpy.ndarray):
                # >>> torch.Tensor(array)
                self._from_numpy(args[0], copy=kwargs.get('copy', True))
            else:
                # >>> torch.Tensor(size)
                if not isinstance(args[0], six.integer_types):
                    raise ValueError('Excepted an integer as size.')
                self._from_shape([args[0]], kwargs.get('dtype', 'float32'))
        else:
            # >>> torch.Tensor(*sizes)
            if not all(isinstance(arg, six.integer_types) for arg in args):
                raise ValueError('Excepted integer(s) as sizes.')
            self._from_shape(args, kwargs.get('dtype', 'float32'))

    @property
    def data(self):
        """Return a data reference detaching the grad.

        Returns
        -------
        dragon.vm.torch.Tensor
            The data tensor.

        """
        return Tensor(device=self.device, id=self._id, own_storage=False)

    @property
    def dtype(self):
        """Return the data type.

        Returns
        -------
        str
            The data type.

        """
        return self._impl.dtype

    @property
    def device(self):
        """Return the device of this tensor.

        Returns
        -------
        dragon.vm.torch.device
           The device.

        """
        return self._device.copy()

    @property
    def grad(self):
        """Return a grad reference if gradient had be computed.

        Returns
        -------
        dragon.vm.torch.Tensor
            The grad tensor.

        """
        grad_id = self._id + '_grad'
        grad_impl = workspace.get_workspace().GetTensor(grad_id)
        if grad_impl is None:
            return None
        grad_ref = Tensor(own_storage=False)
        grad_ref._device = cpp.device(*self._impl.device)
        grad_ref._id, grad_ref._impl = grad_id, grad_impl
        return grad_ref

    @property
    def grad_fn(self):
        return None

    @property
    def id(self):
        """Return the tensor identity.

        Returns
        -------
        str
            The identity.

        """
        return self._id

    @property
    def requires_grad(self):
        """Return a bool report whether the grad is required.

        Returns
        -------
        bool
            **True** if requiring grad otherwise **False**.

        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value
        if self._const_size is not None:
            self._ignored_grads = set() if value \
                else {self._id + '_grad'}

    @property
    def shape(self):
        """Return the shape of this tensor.

        Returns
        -------
        dragon.vm.torch.Size
            The shape.

        """
        return self.size()

    def abs(self):
        r"""Return a tensor with the absolute value.

        .. math:: \text{out} = \left| \text{self} \right|

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.abs(...)`_ : Compute the absolute value of input.

        """
        pass

    def add(self, value):
        r"""Compute the element-wise addition.

        .. math:: \text{out} = \text{self} + \text{value}

        Parameters
        ----------
        value : Union[dragon.vm.torch.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.add(...)`_ : Compute the element-wise addition.

        """
        pass

    def add_(self, value):
        r"""Compute the element-wise addition.

        .. math:: \text{self} \mathrel{+}= \text{value}

        Parameters
        ----------
        value : Union[dragon.vm.torch.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.add(...)`_ : Compute the element-wise addition.

        """
        pass

    def backward(self, gradient=None):
        """Compute the gradients starting from this tensor.

        If ``gradient`` is not provided, **ones** will be used instead.

        Parameters
        ---------
        gradient : dragon.vm.torch.Tensor, optional
            The optional input gradient.

        """
        pass

    def bitwise_not(self):
        r"""Compute the element-wise NOT bitwise operation.

        .. math:: \text{out} = \,\,\sim \text{self}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bitwise_not(...)`_ : Compute the element-wise NOT bitwise operation.

        """
        pass

    def bitwise_not_(self):
        r"""Compute the element-wise NOT bitwise operation.

        .. math:: \text{self} = \,\,\sim \text{self}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bitwise_not(...)`_ : Compute the element-wise NOT bitwise operation.

        """
        pass

    def bitwise_xor(self, other):
        r"""Compute the element-wise XOR bitwise operation.

        .. math:: \text{out} = \text{self} \oplus \text{other}

        Parameters
        ----------
        other : dragon.vm.torch.Tensor
            The second input tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bitwise_xor(...)`_ : Compute the element-wise XOR bitwise operation.

        """
        pass

    def bitwise_xor_(self, other):
        r"""Compute the element-wise XOR bitwise operation.

        .. math:: \text{self} = \text{self} \oplus \text{other}

        Parameters
        ----------
        other : dragon.vm.torch.Tensor
            The second input tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.bitwise_xor(...)`_ : Compute the element-wise XOR bitwise operation.

        """
        pass

    def bool(self):
        """Return a bool tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def bool_(self):
        """Cast to a bool tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def byte(self):
        """Return an uint8 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def byte_(self):
        """Cast to an uint8 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def ceil(self):
        r"""Return a tensor taken the ceil of elements.

        .. math:: \text{out} = \lceil \text{self} \rceil

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.ceil(...)`_ : Compute the smallest integer not less than input.

        """
        pass

    def ceil_(self):
        r"""Set to the ceil of elements.

        .. math:: \text{self} = \lceil \text{self} \rceil

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.ceil(...)`_ : Compute the smallest integer not less than input.

        """
        pass

    def char(self):
        """Return an int8 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def char_(self):
        """Cast to an int8 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def chunk(self, chunks, dim=0):
        """Split self into several parts along the given dim.

        Parameters
        ----------
        chunks : int
            The number of chunks to split.
        dim : int, optional, default=0
            The dim to split.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The output chunks.

        """
        pass

    def clamp(self, min=None, max=None):
        """Return a tensor with elements clamped into a range.

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

        See Also
        --------
        `torch.clamp(...)`_ : Compute the clipped input according to the given bounds.

        """
        pass

    def clamp_(self, min=None, max=None):
        """Clamp elements into the a range.

        Parameters
        ----------
        min : number, optional
            The min value.
        max : number, optional
            The max value.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.clamp(...)`_ : Compute the clipped input according to the given bounds.

        """
        pass

    def copy_(self, src):
        """Copy the elements into this tensor.

        Parameters
        ----------
        src : dragon.vm.torch.Tensor
            The tensor to copy from.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        self._impl.CopyFrom(
            src._impl,
            proto_util.get_device_option(
                self._device.type,
                self._device.index
            ),
            proto_util.get_device_option(
                src._device.type,
                src._device.index
            ),
        )
        # Transfer the const size if necessary
        self._const_size = src.size() \
            if self._const_size else None
        return self

    def cos(self):
        r"""Compute the cos.

        .. math:: \text{out} = \cos(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.cos(...)`_ : Compute the cos of input.

        """
        pass

    def cpu(self):
        """Switch the internal storage on cpu memory.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        self._device.type = 'cpu'
        self._impl.ToCPU()
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
        if device is None:
            cfg = config.config()
            device = cfg.device_index
        self._impl.ToCUDA(device)
        self._device.type, self._device.index = 'cuda', device
        return self

    def cumsum(self, dim):
        """Return a tensor with the cumulative sum of elements.

        Parameters
        ----------
        dim : int
            The cumulative dimension.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.cumsum(...)`_ : Compute the cumulative sum of elements along the given axis.

        """
        pass

    def detach(self):
        """Return a data reference detaching the grad.

        Returns
        -------
        dragon.vm.torch.Tensor
            The data tensor.

        """
        return self.data

    def dim(self):
        """Return the number of dimensions.

        Returns
        -------
        int
            The number of dimensions.

        """
        return self._impl.ndim

    def div(self, value):
        r"""Compute the element-wise division.

        .. math:: \text{out} = \text{self} \div \text{value}

        Parameters
        ----------
        value : Union[dragon.vm.torch.Tensor, number]
            The value to divide.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.div(...)`_ : Compute the element-wise division.

        """
        pass

    def div_(self, value):
        r"""Compute the element-wise division.

        .. math:: \text{self} \mathrel{\div}= \text{value}

        Parameters
        ----------
        value : Union[dragon.vm.torch.Tensor, number]
            The value to be divided.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.div(...)`_ : Compute the element-wise division.

        """
        pass

    def double(self):
        """Return a float64 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def double_(self):
        """Cast to a float64 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def eq(self, other):
        r"""Compute the element-wise equal comparison.

        .. math:: \text{out} = (\text{self} = \text{other})

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.eq(...)`_ : Compute the element-wise equal comparison.

        """
        pass

    def exp(self):
        r"""Compute the exponential.

        .. math:: \text{out} = \exp{\text{self}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.exp(...)`_ : Compute the exponential of input.

        """
        pass

    def expand(self, *sizes):
        """Return a tensor with elements broadcast.

        Parameters
        ----------
        sizes : int...
            The output dimensions to broadcast to.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.expand(...)`_ : Broadcast input according to given sizes.

        """
        pass

    def expand_as(self, other):
        """Return a tensor with elements broadcast like another.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor
            The tensor provided the output dimensions.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.expand(...)`_ : Broadcast input according to given sizes.

        """
        return self.expand(*other.size())

    def fill_(self, value):
        r"""Fill with the given constant value.

        .. math:: \text{self} \leftarrow \text{value}

        Parameters
        ----------
        value : number
            The constant value.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def float(self):
        """Return a float32 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def float_(self):
        """Cast to a float32 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def floor(self):
        r"""Return a tensor taken the floor of elements.

        .. math:: \text{out} = \lfloor \text{self} \rfloor

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.floor(...)`_ : Compute the largest integer not greater than input.

        """
        pass

    def floor_(self):
        r"""Set to the floor of elements.

        .. math:: \text{self} = \lfloor \text{self} \rfloor

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.floor(...)`_ : Compute the largest integer not greater than input.

        """
        pass

    def ge(self, other):
        r"""Compute the element-wise greater-equal comparison.

        .. math:: \text{out} = (\text{self} \geq \text{other})

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.ge(...)`_ : Compute the element-wise greater-equal comparison.

        """
        pass

    def gt(self, other):
        r"""Compute the element-wise greater comparison.

        .. math:: \text{out} = (\text{self} > \text{other})

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.gt(...)`_ : Compute the element-wise greater comparison.

        """
        pass

    def half(self):
        """Return a float16 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def half_(self):
        """Cast to a float16 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def index_select(self, dim, index):
        """Select the elements along the given dim using index.

        Parameters
        ----------
        dim : Union[int, Sequence[int]]
            The dim(s) to select.
        index : dragon.vm.torch.Tensor
            The indices.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def int(self):
        """Return an int32 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def int_(self):
        """Cast to an int32 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def is_floating_point(self):
        """Whether the data type is floating.

        Floating types contains: (*float16*, *float32*, *float64*)

        Returns
        -------
        bool
            **True** if the data type is floating otherwise **False**.

        """
        return 'float' in self.dtype

    def le(self, other):
        r"""Compute the element-wise less-equal comparison.

        .. math:: \text{out} = (\text{self} \leq \text{other})

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.le(...)`_ : Compute the element-wise less-equal comparison.

        """
        pass

    def log(self):
        r"""Compute the natural logarithm.

        .. math:: \text{out} = \log(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def logsumexp(self, dim, keepdim=False):
        r"""Apply the composite of log, sum, and exp.

        .. math:: \text{LogSumExp}(x)_{i} = \log\sum_{j}\exp(x_{ij})

        Parameters
        ----------
        dim : Union[int, Sequence[int]]
            The dimension(s) to reduce.
        keepdim : bool, optional, default=False
            Whether the output tensor has dim retained or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def long(self):
        """Return an int64 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def long_(self):
        """Cast to an int64 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def lt(self, other):
        r"""Compute the element-wise less comparison.

        .. math:: \text{out} = (\text{self} < \text{other})

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.lt(...)`_ : Compute the element-wise less comparison.

        """
        pass

    def masked_fill_(self, mask, value):
        r"""Fill self with the given value where ``mask`` is **1**.

        .. math::
            \text{Ref}[i] =
            \begin{cases}
                \text{Value}[i], & \text{ if } \text{Mask}[i] = 1 \\
                \text{Ref}[i], & \text{ otherwise }
            \end{cases}

        Parameters
        ----------
        mask : dragon.vm.torch.Tensor
            The boolean mask.
        value : number
            The value to fill.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def max(self, dim=None, keepdim=False):
        """Compute the max value of elements along the given axis.

        Parameters
        ----------
        dim : Union[int, Sequence[int]], optional
            The dimension(s) to reduce.
        keepdim : bool, optional, default=False
            Keep the reduced dimensions or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def masked_select(self, mask):
        """Select the elements where mask is **1**.

        Parameters
        ----------
        mask : dragon.vm.torch.Tensor
            The mask for selecting.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def mean(self, dim=None, keepdim=False):
        """Compute the mean value of elements along the given axis.

        Parameters
        ----------
        dim : Union[int, Sequence[int]], optional
            The dimension(s) to reduce.
        keepdim : bool, optional, default=False
            Keep the reduced dimensions or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def min(self, dim=None, keepdim=False):
        """Compute the min value of elements along the given axis.

        Parameters
        ----------
        dim : Union[int, Sequence[int]], optional
            The dimension(s) to reduce.
        keepdim : bool, optional, default=False
            Keep the reduced dimensions or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def mul(self, value):
        r"""Compute the element-wise multiplication.

        .. math:: \text{out} = \text{self} \times \text{value}

        Parameters
        ----------
        value : Union[dragon.vm.torch.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.mul(...)`_ : Compute the element-wise multiplication.

        """
        pass

    def mul_(self, value):
        r"""Compute the element-wise multiplication.

        .. math:: \text{self} \mathrel{\times}= \text{value}

        Parameters
        ----------
        value : Union[dragon.vm.torch.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.mul(...)`_ : Compute the element-wise multiplication.

        """
        pass

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
        pass

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
        pass

    def ndimension(self):
        """Alias for ``Tensor.dim()``.

        Returns
        -------
        int
            The number of dimensions.

        """
        return self.dim()

    def ne(self, other):
        r"""Compute the element-wise not-equal comparison.

        .. math:: \text{out} = (\text{self} \neq \text{other})

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.ne(...)`_ : Compute the element-wise not-equal comparison.

        """
        pass

    def neg(self):
        r"""Compute the element-wise negative.

        .. math:: \text{out} = -\text{self}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.neg(...)`_ : Compute the element-wise negative.

        """
        pass

    def nonzero(self):
        """Return the indices of non-zero elements.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def normal_(self, mean=0, std=1):
        r"""Fill self with a normal distribution.

        .. math:: \text{self} \leftarrow N(\mu, \sigma)

        Parameters
        ----------
        mean : number, optional, default=0
            The value of :math:`\mu`.
        std : number, optional, default=1
            The value of :math:`\sigma`.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def numel(self):
        """Return the total number of elements.

        Returns
        -------
        int
            The total count.

        """
        return self._impl.size

    def numpy(self, readonly=True):
        """Create a numpy array sharing the data.

        Parameters
        ----------
        readonly : bool, optional, default=True
            **False** to sync the content with device.

        Returns
        -------
        numpy.ndarray
            The numpy array.

        """
        return self._impl.ToNumpy(readonly)

    def one_(self):
        r"""Fill with constant 1.

        .. math:: \text{self} \leftarrow 1

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        self.fill_(1)

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
        pass

    def pow(self, exponent):
        """Compute the power.

        Parameters
        ----------
        exponent : Union[dragon.vm.torch.Tensor, number]
            The exponent value.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.pow(...)`_ : Compute the power of input.

        """
        pass

    def reciprocal(self):
        r"""Compute the reciprocal.

        .. math:: \text{out} = \frac{1}{\text{self}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.reciprocal(...)`_ : Compute the reciprocal of input.

        """
        pass

    def reciprocal_(self):
        r"""Compute the reciprocal.

        .. math:: \text{self} = \frac{1}{\text{self}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.reciprocal(...)`_ : Compute the reciprocal of input.

        """
        pass

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
        pass

    def reshape(self, shape):
        """Return a tensor with the same data but a different shape.

        Parameters
        ----------
        shape : Sequence[int]
            The new shape.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.reshape(...)`_ : Change the shape of input.

        """
        pass

    def reshape_(self, shape):
        """Change into a new shape with the same data.

        Parameters
        ----------
        shape : Sequence[int]
            The new shape.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.reshape(...)`_ : Change the shape of input.

        """
        pass

    def round(self):
        r"""Return a tensor taken the round of elements.

        .. math:: \text{out} = \lfloor \text{self} \rceil

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.round(...)`_ : Compute the nearest integer of input.

        """
        pass

    def round_(self):
        r"""Set to the round of elements.

        .. math:: \text{self} = \lfloor \text{self} \rceil

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.round(...)`_ : Compute the nearest integer of input.

        """
        pass

    def rsqrt(self):
        r"""Compute the reciprocal square root.

        .. math:: \text{out} = \frac{1}{\sqrt{\text{self}}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.rsqrt(...)`_ : Compute the square root of input.

        """
        pass

    def rsqrt_(self):
        r"""Compute the reciprocal square root.

        .. math:: \text{self} = \frac{1}{\sqrt{\text{self}}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.rsqrt(...)`_ : Compute the square root of input.

        """
        pass

    def sign(self):
        r"""Return a tensor taken the sign indication of elements.

        .. math::
            \text{out}_{i} =
                \begin{cases}
                    -1, & \text{ if } \text{self}_{i} < 0 \\
                     0, & \text{ if } \text{self}_{i} = 0 \\
                     1, & \text{ if } \text{self}_{i} > 0
                \end{cases}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.sign(...)`_ : Compute the sign indication of input.

        """
        pass

    def sign_(self):
        r"""Set to the sign indication of elements.

        .. math::
            \text{self}_{i} =
                \begin{cases}
                    -1, & \text{ if } \text{self}_{i} < 0 \\
                     0, & \text{ if } \text{self}_{i} = 0 \\
                     1, & \text{ if } \text{self}_{i} > 0
                \end{cases}

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.sign(...)`_ : Compute the sign indication of input.

        """
        pass

    def sin(self):
        r"""Compute the sin.

        .. math:: \text{out} = \sin(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.sin(...)`_ : Compute the sin of input.

        """
        pass

    def size(self, axis=None):
        """Return the size of this tensor.

        Parameters
        ----------
        axis : int, optional
            The optional axis.

        Returns
        -------
        dragon.vm.torch.Size
            The size.

        """
        s = cpp.Size(self._impl.dims)
        return s[axis] if axis is not None else s

    def sqrt(self):
        r"""Compute the square root.

        .. math:: \text{out} = \sqrt{\text{self}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.sqrt(...)`_ : Compute the square root of input.

        """
        pass

    def sqrt_(self):
        r"""Compute the square root.

        .. math:: \text{self} = \sqrt{\text{self}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.sqrt(...)`_ : Compute the square root of input.

        """
        pass

    def squeeze(self, dim=None):
        """Return a tensor with dimensions of size 1 removed.

        Parameters
        ----------
        dim : Union[int, Sequence[int]], optional
            The dimension(s) to remove.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def squeeze_(self, dim=None):
        """Inplace version of ``Tensor.squeeze()``.

        Parameters
        ----------
        dim : Union[int, Sequence[int]], optional
            The dimension(s) to remove.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def sum(self, dim=None, keepdim=False):
        """Compute the sum value of elements along the given axis.

        Parameters
        ----------
        dim : Union[int, Sequence[int]], optional
            The dimension(s) to reduce.
        keepdim : bool, optional, default=False
            Keep the reduced dimensions or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def sub(self, value):
        r"""Compute the element-wise subtraction.

        .. math:: \text{out} = \text{self} - \text{value}

        Parameters
        ----------
        value : Union[dragon.vm.torch.Tensor, number]
            The value to subtract.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.sub(...)`_ : Compute the element-wise subtraction.

        """
        pass

    def sub_(self, value):
        r"""Compute the element-wise subtraction.

        .. math:: \text{self} \mathrel{-}= \text{value}

        Parameters
        ----------
        value : Union[dragon.vm.torch.Tensor, number]
            The value to be subtracted.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.sub(...)`_ : Compute the element-wise subtraction.

        """
        pass

    def type(self, dtype=None):
        """Return the data type.

        If ``dtype`` is not **None**, cast ``self`` to the new tensor.

        Parameters
        ----------
        dtype : str, optional
            The specified type.

        Returns
        -------
        Union[str, dragon.vm.torch.Tensor]
            The data type or new tensor.

        """
        pass

    def uniform_(self, low=0, high=1):
        r"""Fill self with a uniform distribution.

        .. math:: \text{self} \leftarrow U(\alpha, \beta)

        Parameters
        ----------
        low : number, optional, default=0
            The value of :math:`\alpha`.
        high : number, optional, default=1
            The value of :math:`\beta`.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def unsqueeze(self, dim):
        """Return a tensor with dimensions of size 1 inserted.

        Parameters
        ----------
        dim : Union[int, Sequence[int]]
            The dimensions(s) to insert.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def unsqueeze_(self, dim):
        """In-place version of ``Tensor.unsqueeze()``.

        Parameters
        ----------
        dim : Union[int, Sequence[int]]
            The dimensions(s) to insert.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        pass

    def view(self, *shape):
        """Return a tensor with the same data but a different shape.

        Parameters
        ----------
        shape : int...
            The new shape

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.reshape(...)`_ : Change the shape of input.

        """
        return self.reshape(shape)

    def view_(self, *shape):
        """Change into a new shape with the same data.

        Parameters
        ----------
        shape : int...
            The new shape.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.reshape(...)`_ : Change the shape of input.

        """
        return self.reshape_(shape)

    def view_as(self, other):
        """Return a new tensor with the same data
         but a different size as the given tensor.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor
            The tensor to guide the new size.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.reshape(*other.shape)

    def where(self, condition, y):
        r"""Select the elements from two branches under the condition.

        .. math::
            \text{out}[i] =
            \begin{cases}
                \text{self}[i] & \text{ if } cond[i] \text{ is True } \\
                y[i], & \text{ otherwise }
            \end{cases}

        Parameters
        ----------
        condition : dragon.vm.torch.Tensor
            The condition to select one of the branches.
        y : dragon.vm.torch.Tensor
            The tensor :math:`y`.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        pass

    def volatile(self):
        pass

    def zero_(self):
        r"""Fill self with constant 0.

        .. math:: \text{self} \leftarrow 0

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        self.fill_(0)

    def _from_numpy(self, array, copy):
        """Create impl from the numpy array."""
        ws = workspace.get_workspace()
        array = array.copy() if copy else array
        self._const_size = array.size
        self.__gc__ = ws.collectors.TENSOR
        self._id = self.__gc__.alloc(context.get_eager_scope())
        self._impl = ws.CreateTensor(self._id).FromNumpy(array)
        self.requires_grad = self._requires_grad

    def _from_shape(self, shape, dtype):
        """Create impl from the shape and data type."""
        ws = workspace.get_workspace()
        self._const_size = math_util.prod(shape)
        self.__gc__ = ws.collectors.TENSOR
        self._id = self.__gc__.alloc(context.get_eager_scope())
        self._impl = ws.CreateTensor(self._id).FromShape(shape, dtype)
        self.requires_grad = self._requires_grad

    def _type2str(self):
        """Return the tensor type string."""
        return mapping.TENSOR_TYPE_TO_TORCH_TENSOR[self.dtype]

    def __add__(self, other):
        return self.add(other)

    def __del__(self):
        if not self._requires_grad or self._const_size:
            if self._own_storage and self._id:
                # Always reuse the leaf variables or tensors
                # that do not require grad.
                # PyGC will detect them automatically.
                self.__gc__.collect(self._id)

    def __div__(self, other):
        return self.div(other)

    def __float__(self):
        """Return a float python scalar."""
        if self.numel() == 1:
            return float(self.numpy())
        raise TypeError('Only size-1 array can be converted to Python scalars.')

    def __ge__(self, other):
        return self.ge(other)

    def __getitem__(self, item):
        pass

    def __gt__(self, other):
        return self.gt(other)

    def __hash__(self):
        return id(self)

    def __iadd__(self, other):
        return self.add_(other)

    def __idiv__(self, other):
        return self.div_(other)

    def __imul__(self, other):
        return self.mul_(other)

    def __int__(self):
        """Return a int python scalar."""
        return int(self.__float__())

    def __isub__(self, other):
        return self.sub_(other)

    def __itruediv__(self, other):
        return self.div_(other)

    def __le__(self, other):
        return self.le(other)

    def __lt__(self, other):
        return self.lt(other)

    def __mul__(self, other):
        return self.mul(other)

    def __neg__(self):
        return self.neg()

    def __radd__(self, other):
        pass

    def __rdiv__(self, other):
        pass

    def __repr__(self):
        np_data = self.numpy()
        if len(np_data.shape) == 0:
            return str(np_data)
        format_str = str(np_data)
        format_shape = 'x'.join([str(dim) for dim in np_data.shape])
        meta_info = '\n[torch.{} of size {}]'.\
            format(self._type2str(), format_shape)
        if self._device.type == 'cuda':
            meta_info = '\n[torch.cuda.{} of size {} (GPU {})]'.format(
                self._type2str(), format_shape, self._device.index)
        del np_data  # DECREF
        return format_str + meta_info

    def __rmul__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __truediv__(self, other):
        return self.div(other)

    def __setitem__(self, key, value):
        pass

    def __sub__(self, other):
        return self.sub(other)


class BoolTensor(object):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'bool'
        return Tensor(*args, **kwargs)


class ByteTensor(object):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'uint8'
        return Tensor(*args, **kwargs)


class CharTensor(object):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'int8'
        return Tensor(*args, **kwargs)


class DoubleTensor(object):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'float64'
        return Tensor(*args, **kwargs)


class FloatTensor(object):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'float32'
        return Tensor(*args, **kwargs)


class HalfTensor(object):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'float16'
        return Tensor(*args, **kwargs)


class IntTensor(object):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'int32'
        return Tensor(*args, **kwargs)


class LongTensor(object):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'int64'
        return Tensor(*args, **kwargs)


def empty(*sizes, dtype=None, device=None, requires_grad=False):
    """Return a tensor filled with uninitialized data.

    Parameters
    ----------
    sizes : int...
        The sizes of output tensor.
    dtype : str, optional
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device option.
    requires_grad : bool, optional, default=False
        Whether to compute the gradient if necessary.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return Tensor(
        *sizes,
        dtype=dtype if dtype else 'float32',
        device=cpp.device() if device is None else device,
        requires_grad=requires_grad,
    )


def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create a tensor initializing the content from data.

    Parameters
    ----------
    data : array_like
        The data to initialize.
    dtype : str, optional
        The optional data type.
    device : dragon.vm.torch.device, optional
        The optional device of returned tensor.
    requires_grad : bool, optional, default=False
        **True** to record gradient for returned tensor.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    array_data = numpy.array(data, copy=True)
    if dtype is None:
        dtype = str(array_data.dtype)
    else:
        array_data = array_data.astype(dtype)
    return Tensor(
        array_data,
        dtype=dtype,
        device=cpp.device() if device is None else device,
        requires_grad=requires_grad,
    )