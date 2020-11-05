# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.framework import config
from dragon.core.framework import context
from dragon.core.framework import proto_util
from dragon.core.framework import workspace
from dragon.core.util import nest
from dragon.core.util import six
from dragon.core.util import string
from dragon.vm.torch.core import cpp


class Tensor(object):
    """A multi-dimensional array containing elements of a single data type.

    To create a tensor from constant value, use ``torch.tensor(...)``:

    ```python
    # Create a constant tensor
    # The value is 1, dimensions is (0,)
    const_tensor = torch.tensor(1, dtype=torch.float32)
    ```

    Besides, following initializers can also be used:

    ```python
    # Create an empty float32 tensor
    # The dimensions is (1, 2)
    empty_tensor = torch.empty(1, 2, dtype=torch.float32)

    # Create a float32 tensor filling ``one`` or ``zero``
    ones = torch.ones(1, 2, dtype=torch.float32)
    zeros = torch.zeros(1, 2, dtype=torch.float32)
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
        """
        Initialize the device.

        Args:
            self: (todo): write your description
        """
        self._tape = None
        self._is_leaf = False
        self._gc = kwargs.get('gc', None)
        self._impl = kwargs.get('impl', None)
        self._device = kwargs.get('device', cpp.device())
        self._requires_grad = kwargs.get('requires_grad', False)
        if len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                dtype = kwargs.get('dtype', 'float32')
                self._from_numpy(numpy.array(args[0], dtype=dtype), copy=False)
            elif isinstance(args[0], numpy.ndarray):
                self._from_numpy(args[0], copy=kwargs.get('copy', True))
            else:
                if not isinstance(args[0], six.integer_types):
                    raise ValueError('Excepted an integer as size.')
                self._from_shape([args[0]], kwargs.get('dtype', 'float32'))
        elif len(args) > 1:
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
        return Tensor(device=self.device, impl=self._impl)

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
        """Return the grad of this tensor if computed.

        Returns
        -------
        dragon.vm.torch.Tensor
            The grad tensor.

        """
        if self._requires_grad:
            current_ws = workspace.get_workspace()
            impl = current_ws.GetTensor(self.id + '_grad')
            if impl is not None:
                return Tensor(device=self.device, impl=impl)
        return None

    @property
    def grad_fn(self):
        """
        Decorator function that function. n. n.

        Args:
            self: (todo): write your description
        """
        return None

    @property
    def id(self):
        """Return the tensor identity.

        Returns
        -------
        str
            The identity.

        """
        return self._impl.name

    @property
    def is_leaf(self):
        """Return whether tensor is a leaf.

        Returns
        -------
        bool
            **True** if this is a leaf tensor otherwise **False**.

        """
        return self._is_leaf or not self._requires_grad

    @property
    def requires_grad(self):
        """Return whether the grad is required.

        Returns
        -------
        bool
            **True** if requiring grad otherwise **False**.

        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        """
        Set the gradients. gradients.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        self._requires_grad = value

    @property
    def shape(self):
        """Return the shape of this tensor.

        Returns
        -------
        dragon.vm.torch.Size
            The shape.

        """
        return self.size()

    @property
    def volatile(self):
        """Return whether this tensor is volatile.

        Returns
        -------
        bool
            **True** if volatile otherwise **False**.

        """
        return False

    def abs(self):
        r"""Return a tensor with the absolute value.

        .. math:: \text{out} = \left| \text{self} \right|

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.abs(...)`_

        """

    def add(self, other):
        r"""Compute the element-wise addition.

        .. math:: \text{out} = \text{self} + \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.add(...)`_

        """

    def add_(self, other):
        r"""Compute the element-wise addition.

        .. math:: \text{self} \mathrel{+}= \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.add(...)`_

        """

    def argmax(self, dim=None, keepdim=False):
        """Return the index of maximum elements.

        Parameters
        ----------
        dim : int, optional
            The dimension to reduce.
        keepdim : bool, optional, default=False
            Keep the reduced dimension or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The index of maximum elements.

        See Also
        --------
        `torch.argmax(...)`_

        """

    def argmin(self, dim=None, keepdim=False):
        """Return the index of minimum elements.

        Parameters
        ----------
        dim : int, optional
            The dimension to reduce.
        keepdim : bool, optional, default=False
            Keep the reduced dimension or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The index of minimum elements.

        See Also
        --------
        `torch.argmin(...)`_

        """

    def argsort(self, dim=-1, descending=False):
        """Return the index of sorted elements.

        Parameters
        ----------
        dim : int, optional, default=-1
            The dimension to sort elements.
        descending : bool, optional, default=False
            Sort in the descending order or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.argsort(...)`_

        """

    def backward(self, gradient=None, retain_graph=False):
        """Compute the derivatives of this tensor w.r.t. graph leaves.

        Parameters
        ----------
        gradient : dragon.vm.torch.Tensor, optional
            The optional gradient of this tensor.
        retain_graph : bool, optional, default=False
            **False** to free the graph used to compute grad.

        """

    def bitwise_not(self):
        r"""Compute the element-wise NOT bitwise operation.

        .. math:: \text{out} = \,\,\sim \text{self}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bitwise_not(...)`_

        """

    def bitwise_not_(self):
        r"""Compute the element-wise NOT bitwise operation.

        .. math:: \text{self} = \,\,\sim \text{self}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bitwise_not(...)`_

        """

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
        `torch.bitwise_xor(...)`_

        """

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
        `torch.bitwise_xor(...)`_

        """

    def bool(self):
        """Return a bool tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def bool_(self):
        """Cast to a bool tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

    def byte(self):
        """Return an uint8 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def byte_(self):
        """Cast to an uint8 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

    def ceil(self):
        r"""Return a tensor taken the ceil of elements.

        .. math:: \text{out} = \lceil \text{self} \rceil

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.ceil(...)`_

        """

    def ceil_(self):
        r"""Set to the ceil of elements.

        .. math:: \text{self} = \lceil \text{self} \rceil

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.ceil(...)`_

        """

    def char(self):
        """Return an int8 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def char_(self):
        """Cast to an int8 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

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
        `torch.clamp(...)`_

        """

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
        `torch.clamp(...)`_

        """

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
                self._device.index).SerializeToString(),
            proto_util.get_device_option(
                src._device.type,
                src._device.index).SerializeToString(),
        )
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
        `torch.cos(...)`_

        """

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
        """Copy memory to the specified cuda device.

        Parameters
        ----------
        device : Union[int, dragon.vm.torch.device], optional
            The device to copy to.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        if device is None:
            cfg = config.config()
            device = cfg.device_index
        if isinstance(device, cpp.device):
            if device.type != 'cuda':
                raise ValueError('Excepted cuda device, got: ' + device.type)
            device = device.index
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
        `torch.cumsum(...)`_

        """

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

    def div(self, other):
        r"""Compute the element-wise division.

        .. math:: \text{out} = \text{self} \div \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to divide.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.div(...)`_

        """

    def div_(self, other):
        r"""Compute the element-wise division.

        .. math:: \text{self} \mathrel{\div}= \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to be divided.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.div(...)`_

        """

    def double(self):
        """Return a float64 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def double_(self):
        """Cast to a float64 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

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
        `torch.eq(...)`_

        """

    def exp(self):
        r"""Compute the exponential.

        .. math:: \text{out} = \exp(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.exp(...)`_

        """

    def expand(self, *sizes):
        """Return a tensor with elements broadcast.

        Parameters
        ----------
        sizes : Union[Sequence[int], int...]
            The output dimensions to broadcast to.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def expand_as(self, other):
        """Return a tensor with elements broadcast like the other.

        Parameters
        ----------
        other : dragon.vm.torch.Tensor
            The tensor provided the output dimensions.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.expand(*other.size())

    def fill_(self, value):
        r"""Fill self with a scalar value.

        .. math:: \text{self} \leftarrow \text{value}

        Parameters
        ----------
        value : number
            The value to fill.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

    def flatten(self, start_dim=0, end_dim=-1):
        """Return a new tensor with dimensions flattened.

        Parameters
        ----------
        start_dim : int, optional, default=0
            The start dimension to flatten.
        end_dim : int, optional, default=-1
            The end dimension to flatten.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.flatten(...)`_

        """

    def flatten_(self, start_dim=0, end_dim=-1):
        """Flatten the dimensions.

        Parameters
        ----------
        start_dim : int, optional, default=0
            The start dimension to flatten.
        end_dim : int, optional, default=-1
            The end dimension to flatten.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.flatten(...)`_

        """

    def float(self):
        """Return a float32 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def float_(self):
        """Cast to a float32 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

    def floor(self):
        r"""Return a tensor taken the floor of elements.

        .. math:: \text{out} = \lfloor \text{self} \rfloor

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.floor(...)`_

        """

    def floor_(self):
        r"""Set to the floor of elements.

        .. math:: \text{self} = \lfloor \text{self} \rfloor

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.floor(...)`_

        """

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
        `torch.ge(...)`_

        """

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
        `torch.gt(...)`_

        """

    def half(self):
        """Return a float16 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def half_(self):
        """Cast to a float16 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

    def index_select(self, dim, index):
        """Select the elements along the given dim using index.

        Parameters
        ----------
        dim : Union[int, Sequence[int]]
            The dim(s) to select.
        index : dragon.vm.torch.Tensor
            The index.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def int(self):
        """Return an int32 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def int_(self):
        """Cast to an int32 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

    def is_floating_point(self):
        """Return whether the data type is floating.

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
        `torch.le(...)`_

        """

    def log(self):
        r"""Compute the natural logarithm.

        .. math:: \text{out} = \log(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def logsumexp(self, dim, keepdim=False):
        r"""Apply the composite of log, sum, and exp.

        .. math:: \text{out}_{i} = \log\sum_{j}\exp(\text{self}_{ij})

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

    def long(self):
        """Return an int64 tensor with the same data.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def long_(self):
        """Cast to an int64 tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

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
        `torch.lt(...)`_

        """

    def masked_fill_(self, mask, value):
        r"""Fill self with the value where mask is 1.

        .. math::
            \text{self}_{i} =
                \begin{cases}
                    \text{value}_{i}, & \text{ if } \text{mask}_{i} = 1 \\
                    \text{self}_{i}, & \text{ otherwise }
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

    def max(self, dim=None, keepdim=False):
        """Compute the max value of elements along the given dimension.

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

    def mean(self, dim=None, keepdim=False):
        """Compute the mean value of elements along the given dimension.

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

    def min(self, dim=None, keepdim=False):
        """Compute the min value of elements along the given dimension.

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

    def mul(self, other):
        r"""Compute the element-wise multiplication.

        .. math:: \text{out} = \text{self} \times \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.mul(...)`_

        """

    def mul_(self, other):
        r"""Compute the element-wise multiplication.

        .. math:: \text{self} \mathrel{\times}= \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.mul(...)`_

        """

    def multinomial(self, num_samples, epsilon=0):
        """Return a tensor with index sampled from multinomial distribution.

        Parameters
        ----------
        num_samples : int
            The number of samples in each row.
        epsilon : float, optional, default=0
            The epsilon value to apply e-greedy strategy.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

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
        `torch.ne(...)`_

        """

    def neg(self):
        r"""Compute the element-wise negative.

        .. math:: \text{out} = -\text{self}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.neg(...)`_

        """

    def neg_(self):
        r"""Compute the element-wise negative.

        .. math:: \text{self} = -\text{self}

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.neg(...)`_

        """

    def new_empty(
        self,
        *size,
        dtype=None,
        device=None,
        requires_grad=False,
    ):
        """Return a tensor filled with uninitialized data.

        Refer this tensor if ``dtype`` and ``device`` not provided.

        Parameters
        ----------
        size : int...
            The size of output tensor.
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

        See Also
        --------
        `torch.empty(...)`_

        """
        return empty(
            *nest.flatten(size),
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            requires_grad=requires_grad,
        )

    def new_full(
        self,
        size,
        fill_value,
        dtype=None,
        device=None,
        requires_grad=False,
    ):
        """Return a tensor filled with a scalar.

        Refer this tensor if ``dtype`` and ``device`` not provided.

        Parameters
        ----------
        size : Sequence[int]
            The size of output tensor.
        fill_value : number
            The scalar to fill.
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

        See Also
        --------
        `torch.full(...)`_

        """

    def new_ones(
        self,
        *size,
        dtype=None,
        device=None,
        requires_grad=False,
    ):
        """Return a tensor filled with with ones.

        Refer this tensor if ``dtype`` and ``device`` not provided.

        Parameters
        ----------
        size : int...
            The size of output tensor.
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

        See Also
        --------
        `torch.ones(...)`_

        """
        return self.new_full(
            nest.flatten(size),
            fill_value=1,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def new_zeros(
        self,
        *size,
        dtype=None,
        device=None,
        requires_grad=False,
    ):
        """Return a tensor filled with with zeros.

        Refer this tensor if ``dtype`` and ``device`` not provided.

        Parameters
        ----------
        size : int...
            The size of output tensor.
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

        See Also
        --------
        `torch.zeros(...)`_

        """
        return self.new_full(
            nest.flatten(size),
            fill_value=0,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def nonzero(self):
        r"""Return the index of non-zero elements.

        .. math:: \text{out} = \{i\}, \text{ if } \text{self}_{i} \neq 0

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.nonzero(...)`_

        """

    def normal_(self, mean=0, std=1):
        r"""Fill self from a normal distribution.

        .. math:: \text{self} \sim \mathcal{N}(\mu, \sigma^{2})

        Parameters
        ----------
        mean : number, optional, default=0
            The value to :math:`\mu`.
        std : number, optional, default=1
            The value to :math:`\sigma`.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

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
        r"""Fill self with ones.

        .. math:: \text{self} \leftarrow 1

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.fill_(1)

    def permute(self, *dims):
        """Return a new tensor with the specific order of dimensions.

        Parameters
        ----------
        dims : Union[Sequence[int], int...]
            The new order of dimensions.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

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
        `torch.pow(...)`_

        """

    def reciprocal(self):
        r"""Compute the reciprocal.

        .. math:: \text{out} = \frac{1}{\text{self}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.reciprocal(...)`_

        """

    def reciprocal_(self):
        r"""Compute the reciprocal.

        .. math:: \text{self} = \frac{1}{\text{self}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.reciprocal(...)`_

        """

    def repeat(self, *sizes):
        """Repeat elements along the specified dimensions.

        Parameters
        ----------
        sizes : Union[Sequence[int], int...]
            The number of times to repeat.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

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
        `torch.reshape(...)`_

        """

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
        `torch.reshape(...)`_

        """

    def retain_grad(self):
        """Retain grad for the non-leaf tensor."""
        if self._tape:
            self._tape.add_source(self.id)

    def round(self):
        r"""Return a tensor taken the round of elements.

        .. math:: \text{out} = \lfloor \text{self} \rceil

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.round(...)`_

        """

    def round_(self):
        r"""Set to the round of elements.

        .. math:: \text{self} = \lfloor \text{self} \rceil

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.round(...)`_

        """

    def rsqrt(self):
        r"""Compute the reciprocal square root.

        .. math:: \text{out} = \frac{1}{\sqrt{\text{self}}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.rsqrt(...)`_

        """

    def rsqrt_(self):
        r"""Compute the reciprocal square root.

        .. math:: \text{self} = \frac{1}{\sqrt{\text{self}}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.rsqrt(...)`_

        """

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
        `torch.sign(...)`_

        """

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
        `torch.sign(...)`_

        """

    def sin(self):
        r"""Compute the sin.

        .. math:: \text{out} = \sin(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.sin(...)`_

        """

    def size(self, axis=None):
        """Return the size of this tensor.

        Parameters
        ----------
        axis : int, optional
            The optional axis.

        Returns
        -------
        Union[int, dragon.vm.torch.Size]
            The size.

        """
        s = cpp.Size(self._impl.dims)
        return s[axis] if axis is not None else s

    def sort(self, dim=-1, descending=False):
        """Return the sorted elements.

        Parameters
        ----------
        dim : int, optional, default=-1
            The dimension to sort elements.
        descending : bool, optional, default=False
            Sort in the descending order or not.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The value and index tensor.

        See Also
        --------
        `torch.sort(...)`_

        """

    def sqrt(self):
        r"""Compute the square root.

        .. math:: \text{out} = \sqrt{\text{self}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.sqrt(...)`_

        """

    def sqrt_(self):
        r"""Compute the square root.

        .. math:: \text{self} = \sqrt{\text{self}}

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.sqrt(...)`_

        """

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

        See Also
        --------
        `torch.squeeze(...)`_

        """

    def squeeze_(self, dim=None):
        """Remove the dimensions with size 1.

        Parameters
        ----------
        dim : Union[int, Sequence[int]], optional
            The dimension(s) to remove.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.squeeze(...)`_

        """

    def sum(self, dim=None, keepdim=False):
        """Compute the sum value of elements along the given dimension.

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

        See Also
        --------
        `torch.sum(...)`_

        """

    def sub(self, other):
        r"""Compute the element-wise subtraction.

        .. math:: \text{out} = \text{self} - \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to subtract.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.sub(...)`_

        """

    def sub_(self, other):
        r"""Compute the element-wise subtraction.

        .. math:: \text{self} \mathrel{-}= \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to be subtracted.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.sub(...)`_

        """

    def to(self, *args, **kwargs):
        """Convert to the specified data type or device.

        The arguments could be ``torch.dtype`` or ``torch.device``:

        ```python
        x = torch.FloatTensor(1)
        x.to(torch.int32)  # Equivalent to ``x.int()``
        x.to(torch.device('cpu'))  # Equivalent to ``x.cpu()``
        x.to(torch.device('cuda'), torch.float32)  # Convert both
        ```

        Or ``torch.Tensor`` to provide both ``dtype`` and ``device``:

        ```python
        a, b = torch.tensor(1.), torch.tensor(2)
        print(a.to(b))  # 1
        ```

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        dtype = kwargs.get('dtype', None)
        device = kwargs.get('device', None)
        for arg in args:
            if isinstance(arg, cpp.dtype):
                dtype = arg
            elif isinstance(arg, cpp.device):
                device = arg
            elif isinstance(arg, Tensor):
                dtype, device = arg.dtype, arg.device
                break
            else:
                raise ValueError('Unsupported conversion target.')
        if device is not None:
            if device.type == 'cpu':
                self.cpu()
            elif device.type == 'cuda':
                self.cuda(device.index)
            else:
                raise ValueError('Unsupported device type: ' + device.type)
        if dtype is not None:
            return self.type(dtype)
        return self

    def topk(self, k, dim=-1, largest=True, sorted=True):
        """Return the top-K largest or smallest elements.

        Parameters
        ----------
        k : int
            The number of top elements to select.
        dim : int, optional, default=-1
            The dimension to select elements.
        largest : bool, optional
            Return largest or smallest elements.
        sorted : bool, optional
            Whether to return in the sorted order.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The value and index tensor.

        See Also
        --------
        `torch.topk(...)`_

        """

    def transpose(self, dim0, dim1):
        """Return a new tensor with two dimensions swapped.

        Parameters
        ----------
        dim0 : int
            The first dimension to be transposed.
        dim1 : int
            The second dimension to be transposed.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.transpose(...)`_

        """

    def type(self, dtype=None):
        """Return the data type or copied tensor with specified type.

        Parameters
        ----------
        dtype : str, optional
            The specified type to convert to.

        Returns
        -------
        Union[str, dragon.vm.torch.Tensor]
            The data type or copied tensor.

        """

    def uniform_(self, low=0, high=1):
        r"""Fill self from a uniform distribution.

        .. math:: \text{self} \sim \mathcal{U}(\alpha, \beta)

        Parameters
        ----------
        low : number, optional, default=0
            The value to :math:`\alpha`.
        high : number, optional, default=1
            The value to :math:`\beta`.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """

    def unique(self, return_inverse=False, return_counts=False, **kwargs):
        """Return the unique elements.

        Parameters
        ----------
        return_inverse : bool, optional, default=False
            Return the inverse index or not.
        return_counts : bool, optional, default=False
            Return the counts or not.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.
        dragon.vm.torch.Tensor, optional
            The inverse index tensor.
        dragon.vm.torch.Tensor, optional
            The counts tensor.

        See Also
        --------
        `torch.unique(...)`_

        """

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

        See Also
        --------
        `torch.unsqueeze(...)`_

        """

    def unsqueeze_(self, dim):
        """Insert the dimensions of size 1.

        Parameters
        ----------
        dim : Union[int, Sequence[int]]
            The dimensions(s) to insert.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.unsqueeze(...)`_

        """

    def view(self, *shape):
        """Return a tensor with the same data but a different shape.

        Parameters
        ----------
        shape : Union[Sequence[int], int...]
            The new shape

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.reshape(...)`_

        """
        return self.reshape(shape)

    def view_(self, *shape):
        """Change into a new shape with the same data.

        Parameters
        ----------
        shape : Union[Sequence[int], int...]
            The new shape.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        See Also
        --------
        `torch.reshape(...)`_

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
        return self.reshape(other.shape)

    def where(self, condition, y):
        r"""Select the elements from two branches under the condition.

        .. math::
            \text{out}_{i} =
                \begin{cases}
                    \text{self}_{i} & \text{ if } \text{condition}_{i} \\
                    y_{i}, & \text{ otherwise }
                \end{cases}

        Parameters
        ----------
        condition : dragon.vm.torch.Tensor
            The condition tensor.
        y : dragon.vm.torch.Tensor
            The tensor :math:`y`.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.where(...)`_

        """

    def zero_(self):
        r"""Fill self with zeros.

        .. math:: \text{self} \leftarrow 0

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.fill_(0)

    def _from_numpy(self, array, copy):
        """Create impl from the numpy array."""
        ws = workspace.get_workspace()
        array = array.copy() if copy else array
        self._gc, self._is_leaf = ws.collectors.TENSOR, True
        self._impl = ws.create_tensor(self._gc.alloc(
            context.get_eager_scope())).FromNumpy(array)

    def _from_shape(self, shape, dtype):
        """Create impl from the shape and data type."""
        ws = workspace.get_workspace()
        self._gc, self._is_leaf = ws.collectors.TENSOR, True
        self._impl = ws.create_tensor(self._gc.alloc(
            context.get_eager_scope())).FromShape(shape, dtype)

    def __add__(self, other):
        """Compute the element-wise addition.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.add(other)

    def __del__(self):
        """
        Removes the leaf.

        Args:
            self: (todo): write your description
        """
        if self.is_leaf and self._gc:
            # Always reuse the leaf tensors.
            # PyGC will detect them automatically.
            self._gc.collect(self.id)

    def __float__(self):
        """Return a float python scalar.

        Returns
        -------
        float
            The float value.

        """
        return float(self.numpy())

    def __ge__(self, other):
        """Compute the element-wise greater-equal comparison.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.ge(other)

    def __getitem__(self, item):
        """Select elements at the specific index.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def __gt__(self, other):
        """Compute the element-wise greater comparison.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.gt(other)

    def __hash__(self):
        """
        Returns the unique identifier.

        Args:
            self: (todo): write your description
        """
        return id(self)

    def __iadd__(self, other):
        """Compute the element-wise addition.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.add_(other)

    def __imul__(self, other):
        """Compute the element-wise multiplication.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.mul_(other)

    def __int__(self):
        """Return an integer python scalar.

        Returns
        -------
        int
            The integer value.

        """
        return int(self.__float__())

    def __isub__(self, other):
        """Compute the element-wise subtraction.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to be subtracted.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.sub_(other)

    def __itruediv__(self, other):
        """Compute the element-wise division.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to be divided.

        Returns
        -------
        dragon.vm.torch.Tensor
            The self.

        """
        return self.div_(other)

    def __le__(self, other):
        """Compute the element-wise less-equal comparison.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.le(other)

    def __lt__(self, other):
        """Compute the element-wise less comparison.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.lt(other)

    def __mul__(self, other):
        """Compute the element-wise multiplication.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.mul(other)

    def __neg__(self):
        """Compute the element-wise negative.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.neg()

    def __radd__(self, other):
        """Compute the element-wise addition.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def __repr__(self):
        """
        Return a human - readable string representation of this array.

        Args:
            self: (todo): write your description
        """
        array = self.numpy()
        if len(array.shape) == 0:
            return str(array)
        if self._device.type != 'cpu':
            suffix_str = ", device='%s')" % self._device
        else:
            suffix_str = ')'
        debug_str = string.array_to_string(
            array, prefix='tensor(', suffix=suffix_str)
        del array  # DECREF
        return string.add_indent(debug_str, 7)

    def __rmul__(self, other):
        """Compute the element-wise multiplication.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def __rsub__(self, other):
        """Compute the element-wise subtraction.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to be subtracted.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def __rtruediv__(self, other):
        """Compute the element-wise division.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to be divided.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def __truediv__(self, other):
        """Compute the element-wise division.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to divide.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.div(other)

    def __setitem__(self, key, value):
        """Set elements at the specific index."""

    def __sub__(self, other):
        """Compute the element-wise subtraction.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to subtract.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.sub(other)


class BoolTensor(object):
    """The bool tensor."""

    def __new__(cls, *args, **kwargs):
        """
        Create a tensorfluent.

        Args:
            cls: (todo): write your description
        """
        kwargs['dtype'] = 'bool'
        return Tensor(*args, **kwargs)


class ByteTensor(object):
    """The uint8 tensor."""

    def __new__(cls, *args, **kwargs):
        """
        Create a tensorfluent.

        Args:
            cls: (todo): write your description
        """
        kwargs['dtype'] = 'uint8'
        return Tensor(*args, **kwargs)


class CharTensor(object):
    """The int8 tensor."""

    def __new__(cls, *args, **kwargs):
        """
        Create a tensorfluent.

        Args:
            cls: (todo): write your description
        """
        kwargs['dtype'] = 'int8'
        return Tensor(*args, **kwargs)


class DoubleTensor(object):
    """The float64 tensor."""

    def __new__(cls, *args, **kwargs):
        """
        Create a tensorfluent.

        Args:
            cls: (todo): write your description
        """
        kwargs['dtype'] = 'float64'
        return Tensor(*args, **kwargs)


class FloatTensor(object):
    """The float32 tensor."""

    def __new__(cls, *args, **kwargs):
        """
        Create a tensorfluent.

        Args:
            cls: (todo): write your description
        """
        kwargs['dtype'] = 'float32'
        return Tensor(*args, **kwargs)


class HalfTensor(object):
    """The float16 tensor."""

    def __new__(cls, *args, **kwargs):
        """
        Create a tensorfluent.

        Args:
            cls: (todo): write your description
        """
        kwargs['dtype'] = 'float16'
        return Tensor(*args, **kwargs)


class IntTensor(object):
    """The int32 tensor."""

    def __new__(cls, *args, **kwargs):
        """
        Create a tensorfluent.

        Args:
            cls: (todo): write your description
        """
        kwargs['dtype'] = 'int32'
        return Tensor(*args, **kwargs)


class LongTensor(object):
    def __new__(cls, *args, **kwargs):
        """
        Create a tensorfluent.

        Args:
            cls: (todo): write your description
        """
        kwargs['dtype'] = 'int64'
        return Tensor(*args, **kwargs)


def empty(*size, dtype=None, device=None, requires_grad=False):
    """Return a tensor filled with uninitialized data.

    Parameters
    ----------
    size : int...
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
        *size,
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
