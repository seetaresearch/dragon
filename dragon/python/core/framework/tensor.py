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

from dragon.core.framework import context
from dragon.core.framework import device_spec
from dragon.core.framework import types
from dragon.core.framework import workspace
from dragon.core.util import math_util
from dragon.core.util import string


class Tensor(types.TensorBase):
    """A multi-dimensional array for computation."""

    def __init__(
        self,
        shape=None,
        dtype='float32',
        name=None,
        symbolic=False,
        **kwargs
    ):
        """Create a ``Tensor``.

        Parameters
        ----------
        shape : Sequence[int], optional
            The tensor shape.
        dtype : str, optional, default='float32'
            The data type.
        name : str, optional
            The tensor name.
        symbolic : bool, optional, default=False
            Whether to initialize as a symbolic tensor.

        """
        self._shape = None if shape is None else tuple(shape)
        self._dtype = None if dtype is None else str(dtype)
        self._is_variable = not symbolic
        self._impl = kwargs.get('impl', None)
        self._deleter = kwargs.get('deleter', None)
        self._tape = None
        self._grad = None
        self._grad_tape = None
        self._requires_grad = False
        if self._impl is None:
            default_ws = workspace.get_workspace()
            if self._is_variable:
                if self._shape is None or None in self._shape:
                    raise ValueError('Excepted the certain shape to create data.')
                var_scope = context.get_variable_scope()
                self._impl = default_ws.create_tensor(scope=var_scope)
                self._impl.FromShape(self._shape, self._dtype)
                self._deleter = default_ws._handle_pool
            else:
                self._impl = default_ws.create_tensor(scope='Tensor')
                self._deleter = None
        self._name = context.get_name_scope() + name if name else None

    @property
    def device(self):
        """Return the tensor device.

        Returns
        -------
        dragon.DeviceSpec
            The device.

        """
        return device_spec.DeviceSpec(*self._impl.device)

    @property
    def dtype(self):
        """Return the data type.

        Returns
        -------
        str
            The data type.

        """
        if self._is_variable:
            return self._impl.dtype
        return self._dtype

    @property
    def id(self):
        """Return the tensor identity.

        Returns
        -------
        str
            The tensor identity.

        """
        return self._impl.name if self._impl else None

    @property
    def name(self):
        """Return the tensor name.

        Returns
        -------
        str
            The tensor name.

        """
        return self._name

    @property
    def ndim(self):
        """Return the number of dimensions.

        Returns
        -------
        int
            The number of dimensions.

        """
        return len(self.shape)

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
        """Return the tensor shape.

        Returns
        -------
        Tuple[int]
            The tensor shape.

        """
        if self._is_variable:
            return tuple(self._impl.dims)
        return self._shape

    @property
    def size(self):
        """Return the total number of elements in this tensor.

        Returns
        -------
        int
            The total count of elements.

        """
        if self._is_variable:
            return self._impl.size
        if self._shape is None:
            return 0
        if None in self._shape:
            return float('inf')
        return math_util.prod(self._shape)

    @property
    def T(self):
        """Return a tensor with axes reversed.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        return self.transpose()

    def astype(self, dtype, copy=True):
        """Convert the data type to a specific one.

        Parameters
        ----------
        dtype : str
            The specific data type.
        copy : bool, optional, default=True
            Return a new tensor or converted in-place.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.cast(...)`_

        """

    def copy(self):
        """Return a tensor with data copied.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.copy(...)`_

        """

    def fill(self, value):
        r"""Fill self with a scalar value.

        .. math:: \text{self} \leftarrow \text{value}

        Parameters
        ----------
        value : number
            The value to fill.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.fill(...)`_

        """

    def glorot_normal(self, mode='fan_in', scale=2.0):
        r"""Fill self from a glorot normal distribution.

        .. math:: \text{self} \sim \mathcal{N}(0, \frac{scale}{\text{fan}})

        Parameters
        ----------
        mode : {'fan_in, 'fan_out', 'fan_avg'}, optional
            The mode to compute fans.
        scale : float, optional, default=2.0
            The scale factor of distribution.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.random.glorot_normal(...)`_

        """

    def glorot_uniform(self, mode='fan_in', scale=3.0):
        r"""Fill self from a glorot uniform distribution.

        .. math:: \text{self} \sim \mathcal{U}(-\sqrt{\frac{scale}{\text{fan}}},
                                                \sqrt{\frac{scale}{\text{fan}}})

        Parameters
        ----------
        mode : {'fan_in, 'fan_out', 'fan_avg'}, optional
            The mode to compute fans.
        scale : float, optional, default=3.0
            The scale factor of distribution.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.random.glorot_uniform(...)`_

        """

    def item(self):
        """Return the value as a python number.

        Returns
        -------
        number
            The value.

        """
        return float(self) if 'float' in self.dtype else int(self)

    def normal(self, mean=0, std=1):
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
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.random.normal(...)`_

        """

    def numpy(self, copy=False):
        """Convert tensor into a numpy array.

        Parameters
        ----------
        copy : bool, optional, default=False
            Whether to copy the data.

        Returns
        -------
        numpy.ndarray
            The value array.

        """
        return self._impl.ToNumpy(copy)

    def reshape(self, shape, copy=True):
        """Return a tensor containing the same data with new shape.

        Parameters
        ----------
        shape : Union[Sequence[int], dragon.Tensor]
            The output shape.
        copy : bool, optional, default=True
            Return a new tensor or reshape in-place.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.reshape(...)`_

        """

    def tolist(self):
        """Return the value as a python list.

        Returns
        -------
        list
            The value.

        """
        return self.numpy().tolist()

    def transpose(self, perm=None, copy=True):
        """Return a tensor with permuted axes.

        Parameters
        ----------
        perm : Union[Sequence[int], dragon.Tensor]], optional
            The output permutation.
        copy : bool, optional, default=True
            Return a new tensor or transpose in-place.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.transpose(...)`_

        """

    def truncated_normal(self, mean=0, std=1):
        r"""Fill self from a truncated normal distribution.

        .. math:: \text{self} \sim \mathcal{TN}(\mu, \sigma^{2},
                                                \mu - 2\sigma, \mu + 2\sigma)

        Parameters
        ----------
        mean : number, optional, default=0
            The value to :math:`\mu`.
        std : number, optional, default=1
            The value to :math:`\sigma`.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.random.truncated_normal(...)`_

        """

    def uniform(self, low=0, high=1):
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
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.random.uniform(...)`_

        """

    def __add__(self, other):
        """Compute the element-wise addition.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.add(...)`_

        """

    def __and__(self, other):
        """Compute the element-wise AND bitwise operation.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.bitwise.bitwise_and(...)`_

        """

    def __del__(self):
        if self._is_variable and self._deleter:
            self._deleter.release(self._impl.name)

    def __eq__(self, other):
        """Compute element-wise equal comparison.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.equal(...)`_

        """

    def __float__(self):
        """Return the value as a python number.

        Returns
        -------
        float
            The float value.

        """
        return float(self.numpy())

    def __ge__(self, other):
        """Compute element-wise greater-equal comparison.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.greater_equal(...)`_

        """

    def __getitem__(self, item):
        """Select elements at the specific index.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """

    def __gt__(self, other):
        """Compute element-wise greater comparison.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.greater(...)`_

        """

    def __hash__(self):
        """Return the hashable identity."""
        return id(self)

    def __iadd__(self, other):
        """Compute the element-wise addition.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.math.add(...)`_

        """

    def __iand__(self, other):
        """Compute the element-wise AND bitwise operation.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.bitwise.bitwise_and(...)`_

        """

    def __idiv__(self, other):
        """Compute the element-wise division.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to divide.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.math.div(...)`_

        """

    def __imul__(self, other):
        """Compute the element-wise multiplication.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.math.mul(...)`_

        """

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
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.bitwise.invert(...)`_

        """

    def __isub__(self, other):
        """Compute the element-wise subtraction.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to subtract.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.math.sub(...)`_

        """

    def __le__(self, other):
        """Compute element-wise less-equal comparison.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.less_equal(...)`_

        """

    def __lt__(self, other):
        """Compute element-wise less comparison.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.less(...)`_

        """

    def __matmul__(self, other):
        """Compute the matrix multiplication.

        Parameters
        ----------
        other : dragon.Tensor
            The value to multiply.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.matmul(...)`_

        """

    def __mul__(self, other):
        """Compute the element-wise multiplication.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.mul(...)`_

        """

    def __neg__(self):
        """Compute the element-wise negative.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.negative(...)`_

        """

    def __ne__(self, other):
        """Compute element-wise not-equal comparison.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compare.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.not_equal(...)`_

        """

    def __or__(self, other):
        """Compute the element-wise OR bitwise operation.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.bitwise.bitwise_or(...)`_

        """

    def __radd__(self, other):
        """Compute the element-wise addition.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to add.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.add(...)`_

        """

    def __rand__(self, other):
        """Compute the element-wise AND bitwise operation.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.bitwise.bitwise_and(...)`_

        """

    def __repr__(self):
        prefix_str = 'dragon.Tensor('
        main_str = ('shape=(' +
                    ', '.join(['?' if str(dim) == 'None'
                               else str(dim) for dim in self.shape]) +
                    (',)' if len(self.shape) == 1 else ')')) \
            if self.shape is not None else 'shape=None'
        main_str += ', dtype={}'.format(self.dtype)
        if self._is_variable:
            array = self.numpy()
            device = self.device
            main_str = string.array_to_string(
                array, prefix=prefix_str, suffix=', ' + main_str)
            del array
            if device.type != 'cpu':
                main_str += ', device=%s' % str(device)
        else:
            main_str = prefix_str + main_str
        main_str += (')' if self._name is None
                     else ', name="{}")'.format(self._name))
        return string.add_indent(main_str, 14)

    def __rtruediv__(self, other):
        """Compute the element-wise division.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to be divided.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.div(...)`_

        """

    def __rmul__(self, other):
        """Compute the element-wise multiplication.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to multiply.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.mul(...)`_

        """

    def __ror__(self, other):
        """Compute the element-wise OR bitwise operation.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.bitwise.bitwise_or(...)`_

        """

    def __rsub__(self, other):
        """Compute the element-wise subtraction.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to be subtracted.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.sub(...)`_

        """

    def __rxor__(self, other):
        """Compute the element-wise XOR bitwise operation.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.bitwise.bitwise_xor(...)`_

        """

    def __setitem__(self, key, value):
        """Set elements at the specific index."""

    def __sub__(self, other):
        """Compute the element-wise subtraction.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to subtract.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.sub(...)`_

        """

    def __truediv__(self, other):
        """Compute the element-wise division.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to divide.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.math.div(...)`_

        """

    def __xor__(self, other):
        """Compute the element-wise XOR bitwise operation.

        Parameters
        ----------
        other : Union[dragon.Tensor, number]
            The value to compute with.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.bitwise.bitwise_xor(...)`_

        """
