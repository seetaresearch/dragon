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
"""The graph executing tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.framework import context
from dragon.core.framework import types
from dragon.core.framework import workspace
from dragon.core.proto import dragon_pb2
from dragon.core.util import math_util
from dragon.core.util import nest


class Tensor(types.TensorMetaclass):
    """Tensor abstraction for graph executing."""

    def __init__(self, name=None, shape=None, dtype=None):
        """Create a ``Tensor``.

        Parameters
        ----------
        name : str, optional
            The optional tensor name.
        shape : sequence, optional
            The optional tensor shape.
        dtype : str, optional
            The optional data type.

        """
        self._op, self._grad = None, None
        self._name, self._shape, self._dtype = None, None, None
        self.name, self.shape, self.dtype = name, shape, dtype

    @property
    def dtype(self):
        """Return the data type.

        Returns
        -------
        str
            The data type.

        """
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """Set the data type.

        Parameters
        ----------
        value : str
            The data type to set.

        """
        self._dtype = value

    @property
    def id(self):
        """Return the tensor identity.

        Returns
        -------
        str
            The tensor identity.

        """
        return self._name

    @property
    def name(self):
        """Return the tensor name.

        Returns
        -------
        str
            The tensor name.

        """
        return self._name

    @name.setter
    def name(self, value):
        """Set the tensor name.

        Parameters
        ----------
        value : str
            The name to set.

        """
        if value != '':
            value = value if value else 'Tensor'
            name_scope = context.get_name_scope()
            self._name = workspace.get_workspace().unique_name(
                name_scope + value, namespace='Tensor')
        else:
            # Set it manually for same cases
            self._name = value

    @property
    def ndim(self):
        """Return the number of dimensions.

        Returns
        -------
        int
            The number of dimensions.

        """
        if self._shape is not None:
            return len(self._shape)
        return 0

    @property
    def shape(self):
        """Return the tensor shape.

        Returns
        -------
        Sequence[int]
            The shape.

        """
        return self._shape

    @shape.setter
    def shape(self, value):
        """Set the tensor shape.

        Parameters
        ---------
        value : Sequence[int]
            The shape to set.

        """
        if value is not None:
            if not nest.is_sequence(value):
                raise TypeError(
                    'The <shape> should be a Sequence. '
                    'Got {}.'.format(type(value))
                )
            self._shape = nest.flatten(value)
        else:
            self._shape = value

    @property
    def size(self):
        """Return the total number of elements in this tensor.

        Returns
        -------
        int
            The total count of elements.

        """
        if self._shape is None:
            return 0
        if None in self._shape:
            return numpy.inf
        return math_util.prod(self._shape)

    def astype(self, dtype, inplace=False):
        """Cast to the specified data type.

        Parameters
        ----------
        dtype : str
            The data type to cast to.
        inplace : bool, optional, default=False
            Whether to do the cast in-place.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.cast(...)`_ : Cast the data type of input.

        """

    def constant(self, value=0):
        r"""Register self to initialize from a scalar value.

        .. math:: \text{self} \leftarrow \text{value}

        Parameters
        ----------
        value : number, optional, default=0
            The value to initialize.

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('constant', value=value)

    def copy(self):
        """Return a tensor with data copied.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.copy(...)`_ : Copy the input.

        """

    def get_value(self):
        """Return the value of implementation.

        Returns
        -------
        numpy.ndarray
            The deep-copied value.

        """

    def glorot_normal(self, mode='fan_in', scale=2.0):
        r"""Register self to initialize from a glorot uniform distribution.

        .. math:: \text{self} \sim \mathcal{N}(0, \sqrt{\frac{scale}{\text{fan}}})

        Parameters
        ----------
        mode : {'fan_in, 'fan_out', 'fan_avg'}, optional
            The mode to compute fans.
        scale : float, optional, default=2.0
            The scale factor to distribution.

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('glorot_normal', mode=mode, scale=scale)

    def glorot_uniform(self, mode='fan_in', scale=3.0):
        r"""Register self to initialize from a glorot uniform distribution.

        .. math:: \text{self} \sim \mathcal{U}(-\sqrt{\frac{scale}{\text{fan}}},
                                                \sqrt{\frac{scale}{\text{fan}}})

        Parameters
        ----------
        mode : {'fan_in, 'fan_out', 'fan_avg'}, optional
            The mode to compute fans.
        scale : float, optional, default=3.0
            The scale factor to distribution.

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('glorot_uniform', mode=mode, scale=scale)

    def normal(self, mean=0, std=1):
        r"""Register self to initialize from a normal distribution.

        .. math:: \text{self} \sim \mathcal{N}(\mu, \sigma)

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

        """
        return self._register_as('normal', mean=mean, std=std)

    def reshape(self, shape):
        """Return a tensor containing the same data with new shape.

        Parameters
        ----------
        shape : Sequence[int]
            The new shape.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.reshape(...)`_ : Change the dimensions of input.

        """

    def set_value(self, value):
        """Set value to the implementation.

        Parameters
        ----------
        value : array_like
            The value to set.

        Returns
        -------
        dragon.Tensor
            The self.

        """

    def truncated_normal(self, mean=0, std=1):
        r"""Register self to initialize from a truncated normal distribution.

        .. math:: \text{self} \sim \mathcal{TN}(\mu, \sigma, \mu - 2\sigma, \mu + 2\sigma)

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

        """
        return self._register_as('truncated_normal', mean=mean, std=std)

    def uniform(self, low=0, high=1):
        r"""Register self to initialize from an uniform distribution.

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

        """
        return self._register_as('uniform', low=low, high=high)

    @classmethod
    def convert_to(cls, value, dtype=None, name=None):
        """Convert the given ``value`` to a ``dragon.Tensor``.

        Parameters
        ----------
        value : array_like
            The value to convert.
        dtype: str, optional
            The optional data type.
        name: str, optional
            The optional name for this tensor.

        Returns
        -------
        dragon.Tensor
            The constant contains the value.

        """
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value, dtype if dtype else 'float32')
        return TensorRef(
            name=workspace.get_workspace().unique_name(
                name=context.get_name_scope() + (name if name else 'Const'),
                suffix=':0',
                namespace='Tensor'),
            shape=list(value.shape),
            dtype=str(value.dtype),
        ).set_value(value)

    def _register_as(self, type, **kwargs):
        """Fill self with the specific type of filler."""
        filler = dragon_pb2.FillerInfo()
        filler.type = type.lower()
        variance_norm = {'fan_in': 0, 'fan_out': 1, 'fan_avg': 2}
        if filler.type == 'constant':
            filler.value = kwargs['value'] if 'value' in kwargs else 0
        elif filler.type in ['normal', 'gaussian']:
            filler.mean = kwargs['mean'] if 'mean' in kwargs else 0
            filler.std = kwargs['std'] if 'std' in kwargs else 1
            filler.type = 'normal'
        elif filler.type == 'uniform':
            filler.low = kwargs['low'] if 'low' in kwargs else 0
            filler.high = kwargs['high'] if 'high' in kwargs else 1
        elif filler.type == 'truncated_normal':
            filler.mean = kwargs['mean'] if 'mean' in kwargs else 0
            filler.std = kwargs['std'] if 'std' in kwargs else 1
            filler.low = filler.mean - 2.0 * filler.std
            filler.high = filler.mean + 2.0 * filler.std
        elif filler.type in ['glorot_uniform', 'xavier']:
            filler.scale = kwargs['scale'] if 'scale' in kwargs else 3
            filler.variance_norm = variance_norm[kwargs.get('mode', 'fan_in')]
        elif filler.type in ['glorot_normal', 'msra']:
            filler.scale = kwargs['scale'] if 'scale' in kwargs else 2
            filler.variance_norm = variance_norm[kwargs.get('mode', 'fan_in')]
        workspace.get_workspace().create_tensor(self.name, filler)
        return self

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

        """

    def __float__(self):
        """Return a float python scalar.

        Returns
        -------
        float
            The float value.

        """
        return float(self.get_value())

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

        """

    def __getitem__(self, item):
        """Select elements at the specific index.

        Parameters
        ----------
        item : Union[int, slice, dragon.Tensor]
            The index.

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

        """

    def __hash__(self):
        return id(self)

    def __int__(self):
        """Return an integer python scalar.

        Returns
        -------
        int
            The integer value.

        """
        return int(self.get_value())

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

        """

    def __neg__(self):
        """Compute the element-wise negative.

        Returns
        -------
        dragon.Tensor
            The output tensor.

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

        """

    def __repr__(self):
        shape_str = ('(' + ', '.join(
            ['?' if str(dim) == 'None' else str(dim)
                for dim in self.shape]) +
            (',)' if len(self.shape) == 1 else ')')) \
            if self.shape is not None else 'None'
        return 'Tensor("{}", shape={}, dtype={})' \
            .format(self.name, shape_str, self.dtype)

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

        """

    def __setitem__(self, key, value):
        """Set elements at the specific index.

        Parameters
        ----------
        key : Union[int, slice, dragon.Tensor]
            The index.
        value : Union[dragon.Tensor, number]
            The value to set.

        """

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

        """


class TensorRef(object):
    """Create a reference not involved with name scope."""

    def __new__(cls, name, shape=None, dtype=None):
        tensor = Tensor('', shape=shape, dtype=dtype)
        tensor._name = name
        return tensor
