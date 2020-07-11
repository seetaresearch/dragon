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

"""Define the eager tensor abstraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.autograph.tensor import Tensor
from dragon.core.framework import context
from dragon.core.framework import workspace


class EagerTensor(Tensor):
    """Tensor abstraction under the eager execution.

    This abstraction involves the garbage collection,
    thus, life-cycle should be considered to avoid memory leak.

    ```python
    # Create a tensor with shape and dtype
    x = dragon.EagerTensor(shape=(2, 3), dtype='float32')
    y = dragon.EagerTensor(np.ones((2, 3)))

    # Create a tensor copied from the constant
    x = dragon.EagerTensor([2, 3], dtype='float32')
    y = dragon.EagerTensor(np.ones((2, 3)))

    # Create a tensor zero-copied from the array
    x = np.ones((2, 3))
    y = dragon.EagerTensor(x, copy=False)
    ```

    """

    def __init__(self, *args, **kwargs):
        """Create an ``EagerTensor``."""
        super(Tensor, self).__init__()
        self._gc = kwargs.get('gc', None)
        self._impl = kwargs.get('impl', None)
        self._name = kwargs.get('name', None)
        self._device = kwargs.get('device', context.get_device_spec())
        self._requires_grad = kwargs.get('requires_grad', False)
        self._requires_grad = kwargs.get('trainable', self._requires_grad)
        self._is_leaf = False
        if len(args) == 0:
            shape = kwargs.get('shape', None)
            if shape is not None:
                self._from_shape(shape, kwargs.get('dtype', 'float32'))
        elif len(args) == 1:
            self._from_numpy(
                args[0] if isinstance(args[0], numpy.ndarray)
                else numpy.array(args[0], kwargs.get('dtype', 'float32')),
                kwargs.get('copy', True),
            )
        else:
            raise ValueError('Excepted at most one argument.')

    @property
    def device(self):
        """Return the device spec.

        Returns
        -------
        DeviceSpec
           The device spec.

        """
        return self._device.copy()

    @property
    def dtype(self):
        """Return the data type.

        Returns
        -------
        str
            The data type.

        """
        return self._impl.dtype

    @dtype.setter
    def dtype(self, value):
        raise RuntimeError('Call ``astype(...)`` to change the data type.')

    @property
    def id(self):
        """Return the tensor identity.

        Returns
        -------
        str
            The tensor identity.

        """
        return self._impl.name

    @property
    def name(self):
        """Return the tensor name.

        Returns
        -------
        str
            The tensor name.

        """
        return self._name or self._impl.id

    @name.setter
    def name(self, value):
        name_scope = context.get_name_scope()
        self._name = name_scope + (value if value else 'Tensor')

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
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    @property
    def shape(self):
        """Return the shape of this tensor.

        Returns
        -------
        Sequence[int]
            The shape.

        """
        return self._impl.dims

    @shape.setter
    def shape(self, value):
        raise RuntimeError('Call ``reshape(...)`` to change the dimensions.')

    @property
    def size(self):
        """Return the total number of elements in this tensor.

        Returns
        -------
        int
            The total count of elements.

        """
        return self._impl.size

    def astype(self, dtype, inplace=False):
        """Cast the data type to a specific one.

        Parameters
        ----------
        dtype : str
            The specific data type.
        inplace : bool, optional, default=False
            Whether to do the cast in-place.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.cast(...)`_ : Cast the data type of input.

        """

    def constant(self, value=0):
        r"""Fill self with a constant value.

        .. math:: \text{self} \leftarrow \text{value}

        Parameters
        ----------
        value : number, optional, default=0
            The constant value.

        Returns
        -------
        dragon.EagerTensor
            The self.

        """

    def copy(self):
        """Return a tensor with containing data copied.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.copy(...)`_ : Copy the value to ref.

        """

    def get_value(self):
        """Return the value from storage.

        Returns
        -------
        numpy.ndarray
            The shallow copied value.

        """
        return self.numpy()

    def glorot_normal(self, mode='FAN_IN', scale=2.):
        r"""Fill self from a glorot normal distribution.

        .. math:: \text{self} \leftarrow N(0, \sqrt{\frac{scale}{\text{FAN}}})

        Parameters
        ----------
        mode : {'FAN_IN, 'FAN_OUT', 'FAN_AVG'}, optional
            The mode to compute fans.
        scale : number, optional, default=2.
            The scale factor of distribution.

        Returns
        -------
        dragon.EagerTensor
            The self.

        """

    def glorot_uniform(self, mode='FAN_IN', scale=3.):
        r"""Fill self from a glorot uniform distribution.

        .. math:: \text{self} \leftarrow U(
                    -\sqrt{\frac{scale}{\text{FAN}}},
                    \sqrt{\frac{scale}{\text{FAN}}}
                )

        Parameters
        ----------
        mode : {'FAN_IN, 'FAN_OUT', 'FAN_AVG'}, optional
            The mode to compute fans.
        scale : number, optional, default=3.
            The scale factor of distribution.

        Returns
        -------
        dragon.EagerTensor
            The self.

        """

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

    def normal(self, mean=0, std=1):
        r"""Fill self from a normal distribution.

        .. math:: \text{self} \leftarrow N(\mu, \sigma)

        Parameters
        ----------
        mean : number, optional, default=0
            The value of :math:`\mu`.
        std : number, optional, default=1
            The value of :math:`\sigma`.

        Returns
        -------
        dragon.EagerTensor
            The self.

        """

    def reshape(self, shape):
        """Return a tensor containing the same data with new shape.

        Parameters
        ----------
        shape : Sequence[int]
            The new shape.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.reshape(...)`_ : Change the dimensions of input.

        """

    def set_value(self, value):
        """Map the value to storage.

        Parameters
        ----------
        value : array_like
            The value.

        Returns
        -------
        dragon.EagerTensor
            The self.

        """
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value, self.dtype)
        self._impl.FromNumpy(value.copy())
        return self

    def truncated_normal(self, mean=0, std=1):
        r"""Fill self from a truncated normal distribution.

        .. math:: \text{self} \leftarrow TN(\mu, \sigma, \mu - 2\sigma, \mu + 2\sigma)

        Parameters
        ----------
        mean : number, optional, default=0
            The value of :math:`\mu`.
        std : number, optional, default=1
            The value of :math:`\sigma`.

        Returns
        -------
        dragon.EagerTensor
            The self.

        """

    def uniform(self, low=0, high=1):
        self.self__ = r"""Fill self from a uniform distribution.

        .. math:: \text{self} \leftarrow U(\alpha, \beta)

        Parameters
        ----------
        low : number, optional, default=0
            The value of :math:`\alpha`.
        high : number, optional, default=1
            The value of :math:`\beta`.

        Returns
        -------
        dragon.EagerTensor
            The self.

        """

    def _from_numpy(self, array, copy):
        """Create impl from the numpy array."""
        ws = workspace.get_workspace()
        array = array.copy() if copy else array
        self._const_size = array.size
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
        r"""Compute the element-wise addition.

        .. math:: \text{out} = \text{self} + \text{value}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to add.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.add(...)`_ : Compute the element-wise addition.

        """

    def __del__(self):
        if (self._is_leaf or not self._requires_grad) and self._gc:
            # Always reuse the leaf tensors.
            # PyGC will detect them automatically.
            self._gc.collect(self.id)

    def __div__(self, other):
        r"""Compute the element-wise division.

        .. math:: \text{out} = \text{self} \div \text{value}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to divide.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.div(...)`_ : Compute the element-wise division.

        """

    def __float__(self):
        """Return a float python scalar.

        Returns
        -------
        float
            The float value.

        """
        return float(self.numpy())

    def __ge__(self, other):
        r"""Compute element-wise greater-equal comparison.

        .. math:: \text{out} = (\text{self} \geq \text{other})

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to compare.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.greater_equal(...)`_ : Compute element-wise greater-equal comparison.

        """

    def __getitem__(self, item):
        """Select the elements at the specific indices.

        Parameters
        ----------
        item : Union[int, slice, dragon.EagerTensor]
            The indices.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.slice(...)`_ : Select the elements according to the given sections.

        See Also
        --------
        `dragon.masked_select(...)`_ : Select the elements where the given mask is 1.

        """

    def __gt__(self, other):
        r"""Compute element-wise greater comparison.

        .. math:: \text{out} = (\text{self} > \text{other})

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to compare.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.greater(...)`_ : Compute element-wise greater comparison.

        """

    def __hash__(self):
        return id(self)

    def __iadd__(self, other):
        r"""Compute the element-wise addition.

        .. math:: \text{self} \mathrel{+}= \text{other}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to add.

        Returns
        -------
        dragon.EagerTensor
            The self.

        See Also
        --------
        `dragon.math.add(...)`_ : Compute the element-wise addition.

        """

    def __idiv__(self, other):
        r"""Compute the element-wise division.

        .. math:: \text{self} \mathrel{\div}= \text{other}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to divide.

        Returns
        -------
        dragon.EagerTensor
            The self.

        See Also
        --------
        `dragon.math.div(...)`_ : Compute the element-wise division.

        """

    def __imul__(self, other):
        r"""Compute the element-wise multiplication.

        .. math:: \text{self} \mathrel{\times}= \text{other}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to multiply.

        Returns
        -------
        dragon.EagerTensor
            The self.

        See Also
        --------
        `dragon.math.mul(...)`_ : Compute the element-wise multiplication.

        """

    def __int__(self):
        """Return a int python scalar.

        Returns
        -------
        int
            The int value.

        """
        return int(self.__float__())

    def __isub__(self, other):
        r"""Compute the element-wise division.

        .. math:: \text{self} \mathrel{-}= \text{other}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to subtract.

        Returns
        -------
        dragon.EagerTensor
            The self.

        See Also
        --------
        `dragon.math.sub(...)`_ : Compute the element-wise subtraction.

        """

    def __le__(self, other):
        r"""Compute element-wise less-equal comparison.

        .. math:: \text{out} = (\text{self} \leq \text{other})

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to compare.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.less_equal(...)`_ : Compute element-wise less-equal comparison.

        """

    def __lt__(self, other):
        r"""Compute element-wise less comparison.

        .. math:: \text{out} = (\text{self} < \text{other})

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to compare.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.less(...)`_ : Compute element-wise less comparison.

        """

    def __mul__(self, other):
        r"""Compute the element-wise multiplication.

        .. math:: \text{out} = \text{self} \times \text{other}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to multiply.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.mul(...)`_ : Compute the element-wise multiplication.

        """

    def __neg__(self):
        r"""Compute the element-wise negative.

        .. math:: y = -x

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.negative(...)`_ : Compute the element-wise negative.

        """

    def __radd__(self, other):
        r"""Compute the element-wise addition.

        .. math:: \text{out} = \text{other} + \text{self}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to add.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.add(...)`_ : Compute the element-wise addition.

        """

    def __repr__(self):
        array = self.numpy()
        content_str, shape = str(array), array.shape
        del array  # DECREF
        if len(shape) == 0:
            return content_str
        shape_str = ('(' + ', '.join(
            [str(dim) for dim in shape])) + \
            (',)' if len(shape) == 1 else ')')
        meta_str = '\nEagerTensor(shape={}, dtype={}, device={})' \
            .format(shape_str, self.dtype, str(self._device))
        return content_str + meta_str

    def __rdiv__(self, other):
        r"""Compute the element-wise division.

        .. math:: \text{out} = \text{value} \div \text{self}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to be divided.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.div(...)`_ : Compute the element-wise division.

        """

    def __rmul__(self, other):
        r"""Compute the element-wise multiplication.

        .. math:: \text{out} = \text{other} \times \text{self}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to multiply.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.mul(...)`_ : Compute the element-wise multiplication.

        """

    def __rsub__(self, other):
        r"""Compute the element-wise subtraction.

        .. math:: \text{out} = \text{other} - \text{self}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to be subtracted.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.sub(...)`_ : Compute the element-wise subtraction.

        """

    def __setitem__(self, key, value):
        """Set the value at the specific indices.

        Parameters
        ----------
        key : Union[int, slice, dragon.EagerTensor]
            The indices.
        value : number or dragon.EagerTensor
            The value.

        See Also
        --------
        `dragon.assign(...)`_ : Assign the value to ref.

        See Also
        --------
        `dragon.masked_assign(...)`_ : Assign the value to ref where mask is 1.

        """

    def __sub__(self, other):
        r"""Compute the element-wise subtraction.

        .. math:: \text{out} = \text{self} - \text{other}

        Parameters
        ----------
        other : Union[dragon.EagerTensor, number]
            The value to subtract.

        Returns
        -------
        dragon.EagerTensor
            The output tensor.

        See Also
        --------
        `dragon.math.sub(...)`_ : Compute the element-wise subtraction.

        """
