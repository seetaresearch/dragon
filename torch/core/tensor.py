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
"""Tensor class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.framework import config
from dragon.core.framework import context
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
        self._tape = None
        self._device = kwargs.get('device', cpp.device())
        self._impl = kwargs.get('impl', None)
        self._deleter = kwargs.get('deleter', None)
        self._requires_grad = kwargs.get('requires_grad', False)
        self._retains_grad = False
        if len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                dtype = kwargs.get('dtype', 'float32')
                self._from_array(numpy.array(args[0], dtype))
            elif isinstance(args[0], numpy.ndarray):
                dtype = kwargs.get('dtype', None)
                self._from_array(numpy.array(
                    args[0], dtype, copy=kwargs.get('copy', True)))
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
        """Return the grad of this tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The grad tensor.

        """
        if self._requires_grad:
            default_ws = workspace.get_workspace()
            impl = default_ws.get_tensor(self.id + '_grad')
            if impl and impl.size > 0:
                return Tensor(device=self.device, impl=impl)
        return None

    @property
    def grad_fn(self):
        return self._tape

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
            ``True`` if this is a leaf tensor otherwise ``False``.

        """
        return not self._tape or not self._requires_grad

    @property
    def requires_grad(self):
        """Return whether the gradient will be recorded.

        Returns
        -------
        bool
            ``True`` to record gradient otherwise ``False``.

        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
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
    def T(self):
        """Return a tensor with dimensions reversed.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.permute()

    @property
    def volatile(self):
        """Return whether this tensor is volatile.

        Returns
        -------
        bool
            ``True`` if volatile otherwise ``False``.

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
            The output tensor.

        See Also
        --------
        `torch.add(...)`_

        """

    def addmm(self, mat1, mat2, beta=1, alpha=1):
        r"""Add the result of matrix-matrix multiplication.

        .. math:: \text{out} = \alpha (\text{mat1} \times \text{mat2}) + \beta \text{self}

        Parameters
        ----------
        mat1 : dragon.vm.torch.Tensor
            The first matrix.
        mat2 : dragon.vm.torch.Tensor
            The second matrix.
        beta : float, optional, default=1
            The value to :math:`\beta`.
        alpha : float, optional, default=1
            The value to :math:`\alpha`.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.addmm(...)`_

        """

    def argmax(self, dim, keepdim=False):
        """Return the index of maximum elements.

        Parameters
        ----------
        dim : int
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

    def argmin(self, dim, keepdim=False):
        """Return the index of minimum elements.

        Parameters
        ----------
        dim : int
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

    def atan2(self, other):
        r"""Compute the element-wise arc-tangent of two arguments.

        .. math:: \text{out} = \text{arctan}(\frac{\text{self}}{\text{other}})

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
        `torch.atan2(...)`_

        """

    def backward(self, gradient=None, retain_graph=False):
        """Compute the derivatives of this tensor w.r.t. graph leaves.

        Parameters
        ----------
        gradient : dragon.vm.torch.Tensor, optional
            The optional gradient of this tensor.
        retain_graph : bool, optional, default=False
            ``False`` to free the graph used to compute grad.

        """

    def baddbmm(self, batch1, batch2, beta=1, alpha=1):
        r"""Add the result of batched matrix-matrix multiplication.

        .. math::
            \text{out}_{i} = \alpha (\text{batch1}_{i} \times \text{batch2}_{i}) +
                             \beta \text{self}_{i}

        Parameters
        ----------
        batch1 : dragon.vm.torch.Tensor
            The first batch of matrices.
        batch2 : dragon.vm.torch.Tensor
            The second batch of matrices.
        beta : float, optional, default=1
            The value to :math:`\beta`.
        alpha : float, optional, default=1
            The value to :math:`\alpha`.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.baddbmm(...)`_

        """

    def baddbmm_(self, batch1, batch2, beta=1, alpha=1):
        r"""Add the result of batched matrix-matrix multiplication.

        .. math::
            \text{self}_{i} = \alpha (\text{batch1}_{i} \times \text{batch2}_{i}) +
                             \beta \text{self}_{i}

        Parameters
        ----------
        batch1 : dragon.vm.torch.Tensor
            The first batch of matrices.
        batch2 : dragon.vm.torch.Tensor
            The second batch of matrices.
        beta : float, optional, default=1
            The value to :math:`\beta`.
        alpha : float, optional, default=1
            The value to :math:`\alpha`.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.baddbmm(...)`_

        """

    def bitwise_and(self, other):
        r"""Compute the element-wise AND bitwise operation.

        .. math:: \text{out} = \text{self} \mathbin{\&} \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bitwise_and(...)`_

        """

    def bitwise_and_(self, other):
        r"""Compute the element-wise AND bitwise operation.

        .. math:: \text{self} = \text{self} \mathbin{\&} \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bitwise_and(...)`_

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

    def bitwise_or(self, other):
        r"""Compute the element-wise OR bitwise operation.

        .. math:: \text{out} = \text{self} \mathbin{|} \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bitwise_or(...)`_

        """

    def bitwise_or_(self, other):
        r"""Compute the element-wise OR bitwise operation.

        .. math:: \text{self} = \text{self} \mathbin{|} \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bitwise_or(...)`_

        """

    def bitwise_xor(self, other):
        r"""Compute the element-wise XOR logical operation.

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
        `torch.logical_xor(...)`_

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
            The output tensor.

        See Also
        --------
        `torch.bitwise_xor(...)`_

        """

    def bmm(self, batch2):
        r"""Compute the batched matrix multiplication.

        .. math:: \text{out}_{i} = \text{self}_{i} \times \text{batch2}_{i}

        Parameters
        ----------
        batch2 : dragon.vm.torch.Tensor
            The second batch of matrices.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.bmm(...)`_

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
            The output tensor.

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
            The output tensor.

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
            The output tensor.

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
            The output tensor.

        """

    def chunk(self, chunks, dim=0, copy=True):
        """Split self into several parts along the given dim.

        Parameters
        ----------
        chunks : int
            The number of chunks to split.
        dim : int, optional, default=0
            The dimension to split.
        copy : bool, optional, default=True
            Copy or create the views of input.

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
        """Clamp elements into the given range.

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

    def contiguous(self):
        """Return a tensor with contiguous memory.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self

    def copy_(self, src):
        """Copy the elements into this tensor.

        Parameters
        ----------
        src : dragon.vm.torch.Tensor
            The tensor to copy from.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        self._impl.CopyFrom(
            src._impl, self._device.to_proto(), src._device.to_proto())
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
            The output tensor.

        """
        self._impl.ToCPU()
        self._device = cpp.device('cpu')
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
            The output tensor.

        """
        if device is None:
            cfg = config.config()
            device = cfg.device_index
        if isinstance(device, cpp.device):
            if device.type != 'cuda':
                raise ValueError('Excepted cuda device, got: ' + device.type)
            device = device.index
        self._impl.ToCUDA(device)
        self._device = cpp.device('cuda', device)
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
            The output tensor.

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
            The output tensor.

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

    def exp_(self):
        r"""Set to the exponential of elements.

        .. math:: \text{self} = \exp(\text{self})

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
            The output tensor.

        """

    def flatten(self, start_dim=0, end_dim=-1):
        """Return a tensor with dimensions flattened.

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
            The output tensor.

        See Also
        --------
        `torch.flatten(...)`_

        """

    def flip(self, dims):
        """Return a tensor with elements reversed along the given dimension.

        Parameters
        ----------
        dims : Union[int, Sequence[int]]
            The dimension to reverse.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.flip(...)`_

        """

    def fliplr(self):
        """Return a tensor with elements reversed along the second dimension.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.fliplr(...)`_

        """

    def flipud(self):
        """Return a tensor with elements reversed along the first dimension.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.flipud(...)`_

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
            The output tensor.

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
            The output tensor.

        See Also
        --------
        `torch.floor(...)`_

        """

    def gather(self, dim, index):
        """Gather elements along the given dimension of index.

        Parameters
        ----------
        dim : int
            The dimension of index values.
        index : dragon.vm.torch.Tensor
            The index tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.gather(...)`_

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
            The output tensor.

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
            The output tensor.

        """

    def isfinite(self):
        r"""Return if the elements are finite.

        .. math:: \text{out} = \text{isfinite}(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.isfinite(...)`_

        """

    def isinf(self):
        r"""Return if the elements are infinite.

        .. math:: \text{out} = \text{isinf}(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.isinf(...)`_

        """

    def isnan(self):
        r"""Return if the elements are NaN.

        .. math:: \text{out} = \text{isnan}(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.isnan(...)`_

        """

    def is_contiguous(self):
        """Return whether the memory is contiguous.

        Returns
        -------
        bool
            ``True`` if the memory is contiguous otherwise ``False``.

        """
        return True

    def is_floating_point(self):
        """Return whether the data type is floating.

        Floating types contains: (*float16*, *float32*, *float64*)

        Returns
        -------
        bool
            ``True`` if the data type is floating otherwise ``False``.

        """
        return 'float' in self.dtype

    def item(self):
        """Return the value as a python number.

        Returns
        -------
        number
            The value.

        """
        return float(self) if self.is_floating_point() else int(self)

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

        See Also
        --------
        `torch.log(...)`_

        """

    def log_(self):
        r"""Set to the natural logarithm of elements.

        .. math:: \text{self} = \log(\text{self})

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.log(...)`_

        """

    def logical_and(self, other):
        r"""Compute the element-wise AND logical operation.

        .. math:: \text{out} = \text{self} \mathbin{\&} \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.logical_and(...)`_

        """

    def logical_not(self):
        r"""Compute the element-wise NOT logical operation.

        .. math:: \text{out} = \,\,\sim \text{self}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.logical_not(...)`_

        """

    def logical_or(self, other):
        r"""Compute the element-wise OR logical operation.

        .. math:: \text{out} = \text{self} \mathbin{|} \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.logical_or(...)`_

        """

    def logical_xor(self, other):
        r"""Compute the element-wise XOR logical operation.

        .. math:: \text{out} = \text{self} \oplus \text{other}

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.logical_xor(...)`_

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

        See Also
        --------
        `torch.logsumexp(...)`_

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
            The output tensor.

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

    def masked_fill(self, mask, value):
        r"""Return a tensor filled with the value where mask is true.

        .. math::
            \text{out}_{i} =
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
            The output tensor.

        """

    def masked_fill_(self, mask, value):
        r"""Fill self with the value where mask is true.

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
            The output tensor.

        """

    def matmul(self, tensor2):
        r"""Compute the matrix multiplication.

        .. math:: \text{out} = \text{self} \times \text{tensor2}

        Parameters
        ----------
        tensor2 : dragon.vm.torch.Tensor
            The tensor to multiply.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.matmul(...)`_

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

        See Also
        --------
        `torch.max(...)`_

        """

    def maximum(self, other):
        r"""Compute the maximum value of inputs.

        .. math:: \text{out} = \max(\text{self}, \text{other})

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The second input tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.maximum(...)`_

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

        See Also
        --------
        `torch.mean(...)`_

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

        See Also
        --------
        `torch.min(...)`_

        """

    def minimum(self, other):
        r"""Compute the minimum value of inputs.

        .. math:: \text{out} = \min(\text{self}, \text{other})

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The second input tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.minimum(...)`_

        """

    def mm(self, mat2):
        r"""Compute the matrix-matrix multiplication.

        .. math:: \text{out} = \text{self} \times \text{mat2}

        Parameters
        ----------
        mat2 : dragon.vm.torch.Tensor
            The second matrix.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.mm(...)`_

        """

    def mps(self, device=None):
        """Copy memory to the specified mps device.

        Parameters
        ----------
        device : Union[int, dragon.vm.torch.device], optional
            The device to copy to.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        if device is None:
            cfg = config.config()
            device = cfg.device_index
        if isinstance(device, cpp.device):
            if device.type != 'mps':
                raise ValueError('Excepted mps device, got: ' + device.type)
            device = device.index
        self._impl.ToMPS(device)
        self._device = cpp.device('mps', device)
        return self

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
            The output tensor.

        See Also
        --------
        `torch.mul(...)`_

        """

    def multinomial(self, num_samples):
        """Return a tensor with index sampled from multinomial distribution.

        Parameters
        ----------
        num_samples : int
            The number of samples in each row.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def narrow(self, dimension, start, length):
        """Return a narrowed tensor.

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
            The output tensor.

        See Also
        --------
        `torch.neg(...)`_

        """

    def new_empty(self, *size, dtype=None, device=None, requires_grad=False):
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
            ``True`` to record gradient for returned tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.empty(...)`_

        """

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
            ``True`` to record gradient for returned tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.full(...)`_

        """

    def new_ones(self, *size, dtype=None, device=None, requires_grad=False):
        """Return a tensor filled with ones.

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
            ``True`` to record gradient for returned tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.ones(...)`_

        """
        return self.new_full(
            nest.flatten(size), fill_value=1, dtype=dtype, device=device,
            requires_grad=requires_grad)

    def new_tensor(self, data, dtype=None, device=None, requires_grad=False):
        """Return a tensor initializing from the given data.

        Refer this tensor if ``dtype`` and ``device`` not provided.

        Parameters
        ----------
        data : array_like
            The data to initialize from.
        dtype : str, optional
            The optional data type.
        device : dragon.vm.torch.device, optional
            The optional device of returned tensor.
        requires_grad : bool, optional, default=False
            ``True`` to record gradient for returned tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.tensor(...)`_

        """

    def new_zeros(self, *size, dtype=None, device=None, requires_grad=False):
        """Return a tensor filled with zeros.

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
            ``True`` to record gradient for returned tensor.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.zeros(...)`_

        """
        return self.new_full(
            nest.flatten(size), fill_value=0, dtype=dtype, device=device,
            requires_grad=requires_grad)

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

    def norm(self, p='fro', dim=None, keepdim=False, out=None, dtype=None):
        """Compute the norm value of elements along the given dimension.

        Parameters
        ----------
        p : {'fro', 1, 2}, optional
            The norm order.
        dim : Union[int, Sequence[int]], optional
            The dimension to reduce.
        keepdim : bool, optional, default=False
            Keep the reduced dimension or not.
        dtype : str, optional
            The data type to cast to.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.norm(...)`_

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
            The output tensor.

        """

    def numel(self):
        """Return the total number of elements.

        Returns
        -------
        int
            The total count.

        """
        return self._impl.size

    def numpy(self):
        """Create a numpy array sharing the data.

        Returns
        -------
        numpy.ndarray
            The numpy array.

        """
        return self._impl.ToNumpy()

    def one_(self):
        r"""Fill self with ones.

        .. math:: \text{self} \leftarrow 1

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.fill_(1)

    def permute(self, *dims):
        """Return a tensor with the specific order of dimensions.

        Parameters
        ----------
        dims : Union[Sequence[int], int...]
            The new order of dimensions.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """

    def permute_(self, *dims):
        """Reorder the dimensions.

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
            The output tensor.

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

    def reshape(self, *shape):
        """Return a tensor with the same data but a different shape.

        Parameters
        ----------
        shape : Union[Sequence[int], int...]
            The new shape.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.reshape(...)`_

        """

    def reshape_(self, *shape):
        """Change into a new shape with the same data.

        Parameters
        ----------
        shape : Union[Sequence[int], int...]
            The new shape.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.reshape(...)`_

        """

    def retain_grad(self):
        """Retain grad for the non-leaf tensor."""
        if not self._requires_grad:
            raise RuntimeError('Retain grad for a tensor that does not require.')
        self._retains_grad = True

    def roll(self, shifts, dims=None):
        """Return a tensor of rolled elements.

        Parameters
        ----------
        shifts : Union[int, Sequence[int]]
            The rolling offset of each dimension.
        dims : Union[int, Sequence[int]], optional
            The dimension to roll.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.roll(...)`_

        """

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
            The output tensor.

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
            The output tensor.

        See Also
        --------
        `torch.rsqrt(...)`_

        """

    def scatter(self, dim, index, src):
        """Return a tensor with elements updated from the source.

        Parameters
        ----------
        dim : int
            The dimension of index values.
        index : dragon.vm.torch.Tensor
            The index tensor.
        src : Union[dragon.vm.torch.Tensor, number]
            The tensor to update from.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.scatter(...)`_

        """

    def scatter_(self, dim, index, src, reduce=None):
        """Update elements from the source.

        Parameters
        ----------
        dim : int
            The dimension of index values.
        index : dragon.vm.torch.Tensor
            The index tensor.
        src : Union[dragon.vm.torch.Tensor, number]
            The tensor to update from.
        reduce : str, optional
            ``'add'`` or ``'multiply'``.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.scatter(...)`_

        """

    def scatter_add(self, dim, index, src):
        """Return a tensor with elements added from the source.

        Parameters
        ----------
        dim : int
            The dimension of index values.
        index : dragon.vm.torch.Tensor
            The index tensor.
        src : Union[dragon.vm.torch.Tensor, number]
            The tensor to add from.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.scatter_add(...)`_

        """

    def scatter_add_(self, dim, index, src):
        """Add elements from the source.

        Parameters
        ----------
        dim : int
            The dimension of index values.
        index : dragon.vm.torch.Tensor
            The index tensor.
        src : Union[dragon.vm.torch.Tensor, number]
            The tensor to add from.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.scatter_add(...)`_

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
            The output tensor.

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

    def split(self, split_size_or_sections, dim=0, copy=True):
        """Return the split chunks along the given dimension.

        Parameters
        ----------
        split_size_or_sections : Union[int, Sequence[int]
            The number or size of chunks.
        dim : int, optional, default=0
            The dimension to split.
        copy : bool, optional, default=True
            Copy or create the views of input.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The output tensors.

        See Also
        --------
        `torch.split(...)`_

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
            The output tensor.

        See Also
        --------
        `torch.sqrt(...)`_

        """

    def square(self):
        r"""Compute the square of input.

        .. math:: \text{out} = \text{self}^{2}

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.square(...)`_

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
            The output tensor.

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
            The output tensor.

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
            elif device.type == 'mps':
                self.mps(device.index)
            else:
                raise ValueError('Unsupported device type: ' + device.type)
        if dtype is not None:
            return self.type(dtype)
        return self

    def tolist(self):
        """Return the value as a python list.

        Returns
        -------
        list
            The value.

        """
        return self.numpy().tolist()

    def topk(self, k, dim=-1, largest=True, sorted=True):
        """Return the top k-largest or k-smallest elements.

        Parameters
        ----------
        k : int
            The number of top elements to select.
        dim : int, optional, default=-1
            The dimension to select elements.
        largest : bool, optional
            Return largest or smallest elements.
        sorted : bool, optional
            Whether to return elements in the sorted order.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The value and index tensor.

        See Also
        --------
        `torch.topk(...)`_

        """

    def transpose(self, dim0, dim1):
        """Return a tensor with two dimensions swapped.

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

    def transpose_(self, dim0, dim1):
        """Swap two dimensions.

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

    def tril(self, k=0):
        r"""Return the lower triangular part.

        .. math::
            \text{out}_{ij} =
                \begin{cases}
                    0, & \text{ if } j > i + k \\
                    \text{self}_{ij}, & \text{ otherwise }
                \end{cases}

        Parameters
        ----------
        k : int, optional, default=0
            Diagonal above which to zero elements.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.tril(...)`_

        """

    def tril_(self, k=0):
        r"""Set to the lower triangular part.

        .. math::
            \text{self}_{ij} =
                \begin{cases}
                    0, & \text{ if } j > i + k \\
                    \text{self}_{ij}, & \text{ otherwise }
                \end{cases}

        Parameters
        ----------
        k : int, optional, default=0
            Diagonal above which to zero elements.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.tril(...)`_

        """

    def triu(self, k=0):
        r"""Return the upper triangular part.

        .. math::
            \text{out}_{ij} =
                \begin{cases}
                    0, & \text{ if } j < i + k \\
                    \text{self}_{ij}, & \text{ otherwise }
                \end{cases}

        Parameters
        ----------
        k : int, optional, default=0
            Diagonal below which to zero elements.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.triu(...)`_

        """

    def triu_(self, k=0):
        r"""Set to the upper triangular part.

        .. math::
            \text{self}_{ij} =
                \begin{cases}
                    0, & \text{ if } j < i + k \\
                    \text{self}_{ij}, & \text{ otherwise }
                \end{cases}

        Parameters
        ----------
        k : int, optional, default=0
            Diagonal below which to zero elements.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.triu(...)`_

        """

    def type(self, dtype=None):
        """Return the data type.

        If ``dtype`` is not ``None``, converts to a new tensor.

        Parameters
        ----------
        dtype : str, optional
            The data type to convert to.

        Returns
        -------
        Union[str, dragon.vm.torch.Tensor]
            The data type or new tensor.

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
            The output tensor.

        """

    def unbind(self, dim=0, copy=True):
        """Unpack to chunks along the given dimension.

        Parameters
        ----------
        dim : int, optional, default=0
            The dimension to unpack.
        copy : bool, optional, default=True
            Copy or create the views of input.

        Returns
        -------
        Sequence[dragon.vm.torch.Tensor]
            The output tensors.

        See Also
        --------
        `torch.unbind(...)`_

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
            The counting tensor.

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
            The output tensor.

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
        """Change into a new size with the same data.

        Parameters
        ----------
        shape : Union[Sequence[int], int...]
            The new shape.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.reshape(...)`_

        """
        return self.reshape_(shape)

    def view_as(self, other):
        """Return a tensor with the same data but a different size.

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

    def var(self, dim=None, keepdim=False):
        """Compute the variance value of elements along the given dimension.

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
        `torch.var(...)`_

        """

    def zero_(self):
        r"""Fill self with zeros.

        .. math:: \text{self} \leftarrow 0

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.fill_(0)

    def _from_array(self, array):
        """Create implementation from the array."""
        default_ws = workspace.get_workspace()
        var_scope = context.get_variable_scope()
        self._impl = default_ws.create_tensor(scope=var_scope)
        self._impl.FromNumpy(array, False)
        self._deleter = default_ws._handle_pool

    def _from_shape(self, shape, dtype):
        """Create implementation from the shape."""
        default_ws = workspace.get_workspace()
        var_scope = context.get_variable_scope()
        self._impl = default_ws.create_tensor(scope=var_scope)
        self._impl.FromShape(shape, dtype)
        self._deleter = default_ws._handle_pool

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

    def __and__(self, other):
        """Compute the element-wise AND bitwise operation.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.bitwise_and(other)

    def __del__(self):
        if self._deleter:
            self._deleter.release(self._impl.name)

    def __eq__(self, other):
        r"""Compute the element-wise equal comparison.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.eq(other)

    def __float__(self):
        """Return the value as a python number.

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
        """Return the hashable identity."""
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
            The output tensor.

        """
        return self.add_(other)

    def __iand__(self, other):
        """Compute the element-wise AND bitwise operation.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.bitwise_and_(other)

    def __imul__(self, other):
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
        return self.mul_(other)

    def __int__(self):
        """Return the value as a python number.

        Returns
        -------
        int
            The integer value.

        """
        return int(self.numpy())

    def __invert__(self):
        """Compute the element-wise NOT bitwise operation.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.bitwise_not()

    def __ior__(self, other):
        """Compute the element-wise OR bitwise operation.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.bitwise_or_(other)

    def __isub__(self, other):
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
            The output tensor.

        """
        return self.div_(other)

    def __ixor__(self, other):
        """Compute the element-wise XOR bitwise operation.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.bitwise_xor_(other)

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

    def __matmul__(self, other):
        r"""Compute the matrix multiplication.

        .. math:: \text{out} = \text{self} \times \text{tensor2}

        Parameters
        ----------
        other : dragon.vm.torch.Tensor
            The tensor to multiply.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        See Also
        --------
        `torch.matmul(...)`_

        """
        return self.matmul(other)

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

    def __ne__(self, other):
        """Compute the element-wise not-equal comparison.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.ne(other)

    def __neg__(self):
        """Compute the element-wise negative.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.neg()

    def __or__(self, other):
        """Compute the element-wise OR bitwise operation.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.bitwise_or(other)

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
        array = self.numpy()
        if len(array.shape) == 0:
            return str(array)
        if self._device.type != 'cpu':
            suffix_str = ", device='%s')" % self._device
        else:
            suffix_str = ')'
        debug_str = string.array_to_string(
            array, prefix='tensor(', suffix=suffix_str)
        del array
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

    def __xor__(self, other):
        """Compute the element-wise XOR bitwise operation.

        Parameters
        ----------
        other : Union[dragon.vm.torch.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.vm.torch.Tensor
            The output tensor.

        """
        return self.bitwise_xor(other)


class BoolTensor(object):
    """The bool tensor."""

    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'bool'
        return Tensor(*args, **kwargs)


class ByteTensor(object):
    """The uint8 tensor."""

    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'uint8'
        return Tensor(*args, **kwargs)


class CharTensor(object):
    """The int8 tensor."""

    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'int8'
        return Tensor(*args, **kwargs)


class DoubleTensor(object):
    """The float64 tensor."""

    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'float64'
        return Tensor(*args, **kwargs)


class FloatTensor(object):
    """The float32 tensor."""

    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'float32'
        return Tensor(*args, **kwargs)


class HalfTensor(object):
    """The float16 tensor."""

    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'float16'
        return Tensor(*args, **kwargs)


class IntTensor(object):
    """The int32 tensor."""

    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'int32'
        return Tensor(*args, **kwargs)


class LongTensor(object):
    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = 'int64'
        return Tensor(*args, **kwargs)
