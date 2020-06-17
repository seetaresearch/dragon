# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
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

from dragon.core.eager import context
from dragon.core.ops import init_ops
from dragon.core.util import six


class Initializer(object):
    """The base initializer class."""

    @staticmethod
    def _getter(init_fn, **kwargs):
        """Return an named eager tensor."""
        with context.eager_mode():
            value = init_fn(**kwargs)
            value._name = kwargs.get('name', value.id)
        return value

    def __call__(self, shape, dtype='float32', **kwargs):
        """Return a tensor initialized as specified initializer.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the tensor.
        dtype : str, optional, default='float32'
            The optional data type.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        raise NotImplementedError


class Constant(Initializer):
    r"""Fill tensors with a scalar value.

    .. math:: y \leftarrow \text{Value}

    """

    def __init__(self, value=0):
        """Create a ``Constant`` initializer.

        Parameters
        ----------
        value : number, optional, default=0.
            The value fo fill.

        """
        self.value = value

    def __call__(self, shape, dtype='float32', **kwargs):
        return self._getter(
            init_ops.fill,
            value=self.value,
            shape=shape,
            dtype=dtype,
            **kwargs
        )


class GlorotNormal(Initializer):
    r"""Fill tensors according to a glorot normal distribution.

    The **GlorotNormal** distribution is defined as:

    .. math:: X \sim N(0, \sqrt{\frac{\text{scale}}{\text{FAN}}})

    """

    def __init__(self, scale=2., mode='FAN_IN'):
        """Create a ``GlorotNormal`` initializer.

        Parameters
        ----------
        scale : float, optional, default=2.
            The scale factor of distribution.
        mode : {'FAN_IN', 'FAN_OUT', 'FAN_AVG'}, optional
            The mode to compute the normalizer.

        """
        self.scale, self.mode = scale, mode

    def __call__(self, shape, dtype='float32', **kwargs):
        return self._getter(
            init_ops.glorot_normal,
            scale=self.scale,
            mode=self.mode,
            shape=shape,
            dtype=dtype,
            **kwargs
        )


class GlorotUniform(Initializer):
    r"""Fill tensors according to a glorot uniform distribution.

    The **GlorotUniform** distribution is defined as:

    .. math::
        X \sim U(
            -\sqrt{\frac{\text{scale}}{\text{FAN}}},
             \sqrt{\frac{\text{scale}}{\text{FAN}}}
            )

    """

    def __init__(self, scale=3., mode='FAN_IN'):
        """Create a ``GlorotUniform`` initializer.

        Parameters
        ----------
        scale : float, optional, default=3.
            The scale factor of distribution.
        mode : {'FAN_IN', 'FAN_OUT', 'FAN_AVG'}, optional
            The mode to compute the normalizer.

        """
        super(GlorotUniform, self).__init__()
        self.scale, self.mode = scale, mode

    def __call__(self, shape, dtype='float32', **kwargs):
        return self._getter(
            init_ops.glorot_uniform,
            scale=self.scale,
            mode=self.mode,
            shape=shape,
            dtype=dtype,
            **kwargs
        )


class Ones(Initializer):
    r"""Fill tensors with ones.

    .. math:: y \leftarrow 0

    """

    def __init__(self):
        """Create a ``Zeros`` initializer."""
        super(Ones, self).__init__()

    def __call__(self, shape, dtype='float32', **kwargs):
        return self._getter(
            init_ops.fill,
            value=1,
            shape=shape,
            dtype=dtype,
            **kwargs
        )


class RandomNormal(Initializer):
    r"""Fill tensors according to a random normal distribution.

    The **RandomNormal** distribution is defined as:

    .. math:: X \sim N(\mu, \sigma)

    """

    def __init__(self, mean=0., stddev=0.05):
        r"""Create a ``RandomNormal`` initializer.

        Parameters
        ----------
        mean : number, optional, default=0.
            The value of :math:`\mu`.
        stddev : number, optional, default=0.05
            The value of :math:`\sigma`.

        """
        self.mean, self.stddev = mean, stddev

    def __call__(self, shape, dtype='float32', **kwargs):
        return self._getter(
            init_ops.random_normal,
            mean=self.mean,
            std=self.stddev,
            shape=shape,
            dtype=dtype,
            **kwargs
        )


class RandomUniform(Initializer):
    r"""Fill tensors according to a random uniform distribution.

    The **RandomUniform** distribution is defined as:

    .. math:: X \sim U(\alpha, \beta)

    """

    def __init__(self, minval=-0.05, maxval=0.05):
        r"""Create a ``RandomUniform`` initializer.

        Parameters
        ----------
        minval : number, optional, default=-0.05
            The value of :math:`\alpha`.
        maxval : number, optional, default=0.05
            The value of :math:`\beta`.

        """
        self.minval, self.maxval = minval, maxval

    def __call__(self, shape, dtype='float32', **kwargs):
        return self._getter(
            init_ops.random_uniform,
            low=self.minval,
            high=self.maxval,
            shape=shape,
            dtype=dtype,
            **kwargs
        )


class TruncatedNormal(Initializer):
    r"""Fill tensors according to a truncated normal distribution.

    The **TruncatedNormal** distribution is defined as:

    .. math::
        X \sim TN(\mu, \sigma, \mu - 2\sigma, \mu + 2\sigma)

    """

    def __init__(self, mean=0., stddev=0.05):
        r"""Create a ``TruncatedNormal`` initializer.

        Parameters
        ----------
        mean : number, optional, default=0.
            The value of :math:`\mu`.
        stddev : number, optional, default=0.05
            The value of :math:`\sigma`.

        """
        self.mean, self.stddev = mean, stddev

    def __call__(self, shape, dtype='float32', **kwargs):
        return self._getter(
            init_ops.truncated_normal,
            mean=self.mean,
            std=self.stddev,
            shape=shape,
            dtype=dtype,
            **kwargs
        )


class Zeros(Initializer):
    r"""Fill tensors with zeros.

    .. math:: y \leftarrow 1

    """

    def __init__(self):
        """Create a ``Zeros`` initializer."""
        super(Zeros, self).__init__()

    def __call__(self, shape, dtype='float32', **kwargs):
        return self._getter(
            init_ops.fill,
            value=0,
            shape=shape,
            dtype=dtype,
            **kwargs
        )


def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, six.string_types):
        return globals()[identifier]()
    else:
        raise TypeError(
            'Could not interpret initializer identifier: {}.'
            .format(repr(identifier))
        )


# Aliases
zeros = Zeros
ones = Ones
constant = Constant
random_normal = RandomNormal
random_uniform = RandomUniform
truncated_normal = TruncatedNormal
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform
