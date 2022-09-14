# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py>
#
# ------------------------------------------------------------
"""Init ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import constant_ops
from dragon.core.ops import random_ops


class Initializer(object):
    """The basic Initializer."""

    def __call__(self, shape, dtype=None, **kwargs):
        """Return a tensor initialized from the initializer.

        Parameters
        ----------
        shape : Sequence[int]
            The tensor shape.
        dtype : str, optional
            The optional data type.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """


class Constant(Initializer):
    r"""Fill tensor with a scalar value.

    .. math:: \text{tensor} \leftarrow \text{value}

    """

    def __init__(self, value=0, dtype='float32'):
        """Create a ``Constant`` initializer.

        Parameters
        ----------
        value : number, optional, default=0
            The scalar value to fill.
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        self.value, self.dtype = value, dtype

    def __call__(self, shape, dtype=None, **kwargs):
        """Return a tensor initialized from the initializer.

        Parameters
        ----------
        shape : Sequence[int]
            The tensor shape.
        dtype : str, optional
            The optional data type.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        dtype = dtype or self.dtype
        dtype = str(dtype) if dtype else dtype
        return constant_ops.fill(shape, self.value, dtype=dtype)


class RandomNormal(Initializer):
    r"""Fill tensor from a normal distribution.

    .. math:: \text{tensor} \sim \mathcal{N}(\mu, \sigma^{2})

    """

    def __init__(self, mean=0, stddev=1, dtype='float32'):
        r"""Create a ``RandomNormal`` initializer.

        Parameters
        ----------
        mean : number, optional, default=0
            The value to :math:`\mu`.
        stddev : number, optional, default=1
            The value to :math:`\sigma`.
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        self.mean, self.stddev, self.dtype = mean, stddev, dtype

    def __call__(self, shape, dtype=None, **kwargs):
        """Return a tensor initialized from the initializer.

        Parameters
        ----------
        shape : Sequence[int]
            The tensor shape.
        dtype : str, optional
            The optional data type.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        dtype = dtype or self.dtype
        dtype = str(dtype) if dtype else dtype
        return random_ops.random_normal(
            shape, self.mean, self.stddev, dtype=dtype)


class RandomUniform(Initializer):
    r"""Fill tensor from an uniform distribution.

    .. math:: \text{tensor} \sim \mathcal{U}(\alpha, \beta)

    """

    def __init__(self, minval=0, maxval=1, dtype='float32'):
        r"""Create a ``RandomUniform`` initializer.

        Parameters
        ----------
        minval : number, optional, default=0
            The value to :math:`\alpha`.
        maxval : number, optional, default=1
            The value to :math:`\beta`.
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        self.minval, self.maxval, self.dtype = minval, maxval, dtype

    def __call__(self, shape, dtype=None, **kwargs):
        """Return a tensor initialized from the initializer.

        Parameters
        ----------
        shape : Sequence[int]
            The tensor shape.
        dtype : str, optional
            The optional data type.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        dtype = dtype or self.dtype
        dtype = str(dtype) if dtype else dtype
        return random_ops.random_uniform(
            shape, self.minval, self.maxval, dtype=dtype)


class TruncatedNormal(Initializer):
    r"""Fill tensor from a truncated normal distribution.

    .. math:: \text{tensor} \sim \mathcal{TN}(\mu, \sigma^{2}, \mu - 2\sigma, \mu + 2\sigma)

    """

    def __init__(self, mean=0, stddev=1, dtype='float32'):
        r"""Create a ``TruncatedNormal`` initializer.

        Parameters
        ----------
        mean : number, optional, default=0
            The value to :math:`\mu`.
        stddev : number, optional, default=1
            The value to :math:`\sigma`.
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        self.mean, self.stddev, self.dtype = mean, stddev, dtype

    def __call__(self, shape, dtype=None, **kwargs):
        """Return a tensor initialized from the initializer.

        Parameters
        ----------
        shape : Sequence[int]
            The tensor shape.
        dtype : str, optional
             The optional data type.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        dtype = dtype or self.dtype
        dtype = str(dtype) if dtype else dtype
        return random_ops.truncated_normal(
            shape, self.mean, self.stddev, dtype=dtype)


class VarianceScaling(Initializer):
    """Fill tensor from a scaled random distribution."""

    def __init__(
        self,
        scale=1.0,
        mode='fan_out',
        distribution='normal',
        dtype='float32',
    ):
        """Create a ``RandomNormal`` initializer.

        Parameters
        ----------
        scale : float, optional, default=1
            The scale factor to distribution.
        mode : {'fan_in', 'fan_out', 'fan_avg'}, optional
            The mode for adapting to shape.
        distribution : {'normal', 'uniform'}, optional
            The optional distribution to generate values.
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        if scale <= 0.:
            raise ValueError('<scale> must be positive float.')
        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid <mode> argument: ' + mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError("Invalid `distribution` argument:", distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        """Return a tensor initialized from the initializer.

        Parameters
        ----------
        shape : Sequence[int]
            The tensor shape.
        dtype : str, optional
             The optional data type.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        dtype = dtype or self.dtype
        dtype = str(dtype) if dtype else dtype
        if self.distribution == 'normal':
            return random_ops.glorot_normal(
                shape, mode=self.mode, scale=self.scale * 2.0, dtype=dtype)
        else:
            return random_ops.glorot_uniform(
                shape, mode=self.mode, scale=self.scale * 3.0, dtype=dtype)


class GlorotNormal(VarianceScaling):
    r"""Fill tensor from a glorot normal distribution.

    .. math:: \text{tensor} \sim \mathcal{N}(0, \frac{2}{\text{fan\_avg}})

    """

    def __init__(self, dtype='float32'):
        """Create a ``GlorotNormal`` initializer.

        Parameters
        ----------
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        super(GlorotNormal, self).__init__(
            scale=1.0, mode='fan_out', distribution='normal', dtype=dtype)


class GlorotUniform(VarianceScaling):
    r"""Fill tensor from a glorot uniform distribution.

    .. math:: \text{tensor} \sim \mathcal{U}(-\sqrt{\frac{3}{\text{fan\_avg}}},
                                              \sqrt{\frac{3}{\text{fan\_avg}}})

    """

    def __init__(self, dtype='float32'):
        """Create a ``GlorotUniform`` initializer.

        Parameters
        ----------
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        super(GlorotUniform, self).__init__(
            scale=1., mode='fan_out', distribution='uniform', dtype=dtype)


class Ones(Initializer):
    r"""Fill tensor with ones.

    .. math:: \text{tensor} \leftarrow 1

    """

    def __init__(self, dtype='float32'):
        """Create a ``Ones`` initializer.

        Parameters
        ----------
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        """Return a tensor initialized from the initializer.

        Parameters
        ----------
        shape : Sequence[int]
            The tensor shape.
        dtype : str, optional
            The optional data type.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        dtype = str(self.dtype) if dtype is None else dtype
        return constant_ops.fill(shape, value=1, dtype=dtype)


class Zeros(Initializer):
    r"""Fill tensor with zeros.

    .. math:: \text{tensor} \leftarrow 0

    """

    def __init__(self, dtype='float32'):
        """Create a ``Zeros`` initializer.

        Parameters
        ----------
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        self.dtype = dtype

    def __call__(self, shape, dtype=None, **kwargs):
        """Return a tensor initialized from the initializer.

        Parameters
        ----------
        shape : Sequence[int]
            The tensor shape.
        dtype : str, optional
            The optional data type.

        Returns
        -------
        dragon.Tensor
            The output tensor.

        """
        dtype = str(self.dtype) if dtype is None else dtype
        return constant_ops.fill(shape, value=0, dtype=dtype)


# Aliases
zeros_initializer = zero = zeros = Zeros
ones_initializer = one = ones = Ones
constant_initializer = constant = Constant
random_uniform_initializer = uniform = random_uniform = RandomUniform
random_normal_initializer = normal = random_normal = RandomNormal
truncated_normal_initializer = truncated_normal = TruncatedNormal
variance_scaling_initializer = VarianceScaling
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform
