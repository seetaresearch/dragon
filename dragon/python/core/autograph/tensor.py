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

"""Define the symbolic tensor abstraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from dragon.core.framework import context
from dragon.core.framework import types
from dragon.core.framework import workspace
from dragon.core.proto import dragon_pb2
from dragon.core.util import nest


class Tensor(types.TensorMetaclass):
    """Tensor abstraction under the graph execution.

    It is provided to construct operators symbolically,
    while can also be a navigation to the storage.

    """

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
        self._op = None
        self._grad = None
        self.name = name
        self.shape = shape
        self.dtype = dtype

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
            self._name = workspace.get_dummy_name(
                name_scope + value, domain='Tensor')
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
        if not hasattr(self, '_shape'):
            self._shape = None
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
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.cast(...)`_ : Cast the data type of input.

        """
        pass

    def constant(self, value=0):
        r"""Register as a variable with constant initializer.

        .. math:: \text{self} \leftarrow \text{value}

        Parameters
        ----------
        value : number, optional, default=0
            The constant value.

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('constant', value=value)

    def copy(self):
        """Return a tensor with containing data copied.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        See Also
        --------
        `dragon.copy(...)`_ : Copy the value to ref.

        """
        pass

    def get_value(self):
        """Copy the data from storage.

        Returns
        -------
        numpy.ndarray
            The deep copied value.

        See Also
        --------
        `dragon.workspace.fetch_tensor(...)`_ : Fetch the value of given tensor.

        """
        pass

    def glorot_normal(self, scale=2.):
        r"""Register as a variable with glorot normal initializer.

        .. math:: \text{self} \leftarrow N(0, \sqrt{\frac{scale}{\text{FanIn}}})

        Parameters
        ----------
        scale : number, optional, default=2.
            The scale factor of distribution.

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('glorot_normal', scale=scale)

    def glorot_uniform(self, scale=3.):
        r"""Register as a variable with glorot uniform initializer.

        .. math:: \text{self} \leftarrow U(
                -\sqrt{\frac{scale}{\text{FanIn}}},
                \sqrt{\frac{scale}{\text{FanIn}}}
            )

        Parameters
        ----------
        scale : number, optional, default=3.
            The scale factor of distribution.

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('glorot_uniform', scale=scale)

    def normal(self, mean=0, std=1):
        r"""Register as a variable with normal initializer.

        .. math:: \text{self} \leftarrow N(\mu, \sigma)

        Parameters
        ----------
        mean : number, optional, default=0
            The value of :math:`\mu`.
        std : number, optional, default=1
            The value of :math:`\sigma`.

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('normal', mean=mean, std=std)

    def placeholder(self):
        r"""Register as a placeholder with zero initializer.

        .. math:: \text{self} \leftarrow 0

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('placeholder')

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
        pass

    def set_value(self, value):
        """Feed the const value to the storage.

        Parameters
        ----------
        value : array_like
            The const value.

        Returns
        -------
        dragon.Tensor
            The self.

        See Also
        --------
        `dragon.workspace.feed_tensor(...)`_ : Feed the value to the given tensor.

        """
        pass

    def truncated_normal(self, mean=0, std=1):
        r"""Register as a variable with truncated normal initializer.

        .. math:: \text{self} \leftarrow TN(\mu, \sigma, \mu - 2\sigma, \mu + 2\sigma)

        Parameters
        ----------
        mean : number, optional, default=0
            The value of :math:`\mu`.
        std : number, optional, default=1
            The value of :math:`\sigma`.

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('truncated_normal', mean=mean, std=std)

    def uniform(self, low=0, high=1):
        r"""Register as a variable with uniform initializer.

        .. math:: \text{self} \leftarrow U(\alpha, \beta)

        Parameters
        ----------
        low : number, optional, default=0
            The value of :math:`\alpha`.
        high : number, optional, default=1
            The value of :math:`\beta`.

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('uniform', low=low, high=high)

    def variable(self):
        r"""Register as a variable with zero initializer.

        .. math:: self \leftarrow 0

        Returns
        -------
        dragon.Tensor
            The self.

        """
        return self._register_as('variable')

    @classmethod
    def convert_to(cls, value, dtype=None, name=None):
        """Convert the given ``value`` to a ``dragon.Tensor``.

        Parameters
        ----------
        value : Union[number, Sequence, numpy.ndarray]
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
        return RefTensor('', dtype=dtype)._from_constant(value, name)

    def _register_as(self, type, **kwargs):
        """Fill self with the specific type of filler."""
        filler = dragon_pb2.TensorFillerProto()
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
        workspace.create_filler(filler)
        return self

    def _from_constant(self, value, name=None):
        """Convert the value to a tensor."""
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value, self.dtype if self.dtype else 'float32')
        return RefTensor(
            name=workspace.get_dummy_name(
                basename=context.get_name_scope() +
                        (name if name else 'Const'),
                suffix=':0',
                domain='Tensor'
            ),
            shape=list(value.shape),
            dtype=str(value.dtype),
        ).set_value(value)

    def __add__(self, other):
        pass

    def __div__(self, other):
        pass

    def __float__(self):
        """Return a float python scalar.

        Returns
        -------
        float
            The float value.

        """
        return float(self.get_value())

    def __ge__(self, other):
        pass

    def __getitem__(self, item):
        pass

    def __gt__(self, other):
        pass

    def __hash__(self):
        return id(self)

    def __int__(self):
        """Return a int python scalar.

        Returns
        -------
        int
            The int value.

        """
        return int(self.get_value())

    def __lt__(self, other):
        pass

    def __le__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __neg__(self):
        pass

    def __radd__(self, other):
        pass

    def __repr__(self):
        shape_str = ('(' + ', '.join(
            ['?' if str(dim) == 'None' else str(dim)
                for dim in self.shape]) +
            (',)' if len(self.shape) == 1 else ')')) \
            if self.shape is not None else 'None'
        return 'Tensor("{}", shape={}, dtype={})' \
            .format(self.name, shape_str, self.dtype)

    def __rdiv__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __rtruediv__(self, other):
        return self.__div__(other)

    def __setitem__(self, key, value):
        pass

    def __sub__(self, other):
        pass

    def __truediv__(self, other):
        return self.__div__(other)


class RefTensor(object):
    """Create a reference tensor not involved with name scope."""

    def __new__(cls, name, shape=None, dtype=None):
        tensor = Tensor('', shape=shape, dtype=dtype)
        tensor._name = name
        return tensor
