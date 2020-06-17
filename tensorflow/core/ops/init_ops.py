# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import init_ops


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
        raise NotImplementedError


class Constant(Initializer):
    """Fill tensors with a scalar."""

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
        dtype = str(self.dtype) if dtype is None else str(dtype)
        return init_ops.fill(shape, value=self.value, dtype=dtype)


class Ones(Initializer):
    """Fill tensors with ones."""

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
        dtype = str(self.dtype) if dtype is None else str(dtype)
        return init_ops.fill(shape, value=1, dtype=str(dtype))


class RandomUniform(Initializer):
    """Fill tensors according to a uniform distribution."""

    def __init__(self, minval=0, maxval=1, dtype='float32'):
        """Create a ``RandomUniform`` initializer.

        Parameters
        ----------
        minval : number, optional, default=0
            The lower bound of distribution.
        maxval : number, optional, default=1
            The higher bound of distribution.
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
        dtype = str(self.dtype) if dtype is None else str(dtype)
        return init_ops.random_uniform(
            shape=shape,
            low=self.minval,
            high=self.maxval,
            dtype=str(self.dtype) if dtype is None else str(dtype),
        )


class RandomNormal(Initializer):
    """Fill tensors according to a normal distribution."""

    def __init__(self, mean=0, stddev=1, dtype='float32'):
        """Create a ``RandomNormal`` initializer.

        Parameters
        ----------
        mean : number, optional, default=0
            The mean of distribution.
        stddev : number, optional, default=1
            The stddev of distribution.
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
        return init_ops.random_normal(
            shape=shape,
            mean=self.mean,
            std=self.stddev,
            dtype=str(self.dtype) if dtype is None else str(dtype),
        )


class TruncatedNormal(Initializer):
    r"""Fill tensors according to a truncated normal distribution.

    The **TruncatedNormal** distribution is defined as:

    .. math::
        X \sim TN(\mu, \sigma, \mu - 2\sigma, \mu + 2\sigma)

    """

    def __init__(self, mean=0, stddev=1, dtype='float32'):
        """Create a ``TruncatedNormal`` initializer.

        Parameters
        ----------
        mean : number, optional, default=0
            The mean of distribution.
        stddev : number, optional, default=1
            The stddev of distribution.
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
        return init_ops.truncated_normal(
            shape=shape,
            mean=self.mean,
            std=self.stddev,
            dtype=str(self.dtype) if dtype is None else str(dtype),
        )


class VarianceScaling(Initializer):
    """Fill tensors with the random values adapting to shape."""

    def __init__(
        self,
        scale=1.,
        mode='fan_in',
        distribution='normal',
        dtype='float32',
    ):
        """Create a ``RandomNormal`` initializer.

        Parameters
        ----------
        scale : float, optional, default=1.
            The scale factor applied to distribution.
        mode : {'fan_in', 'fan_out', 'fan_avg'}, optional
            The mode for adapting to shape.
        distribution : {'normal', 'uniform'}, optional
            The optional distribution to generate values.
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        if scale <= 0.:
            raise ValueError("`scale` must be positive float.")

        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError("Invalid `mode` argument:", mode)

        distribution = distribution.lower()
        if distribution not in {"normal", "uniform"}:
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
        if self.distribution == 'normal':
            return init_ops.glorot_normal(
                shape=shape,
                scale=self.scale * 2.,
                mode=self.mode,
                dtype=str(self.dtype) if dtype is None else str(dtype)
            )
        else:
            return init_ops.glorot_uniform(
                shape=shape,
                scale=self.scale * 3.,
                mode=self.mode,
                dtype=str(self.dtype) if dtype is None else str(dtype)
            )


class GlorotNormal(VarianceScaling):
    """Fill tensors according to a glorot normal distribution."""

    def __init__(self, dtype='float32'):
        """Create a ``GlorotNormal`` initializer.

        Parameters
        ----------
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        super(GlorotNormal, self).__init__(
            scale=1.,
            mode='fan_avg',
            distribution='normal',
            dtype=dtype,
        )


class GlorotUniform(VarianceScaling):
    """Fill tensors according to a glorot uniform distribution."""

    def __init__(self, dtype='float32'):
        """Create a ``GlorotUniform`` initializer.

        Parameters
        ----------
        dtype : str, optional, default='float32'
            The data type to set as default.

        """
        super(GlorotUniform, self).__init__(
            scale=1.,
            mode='fan_avg',
            distribution='uniform',
            dtype=dtype,
        )


class Zeros(Initializer):
    """Fill tensors with zeros."""

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
        dtype = str(self.dtype) if dtype is None else str(dtype)
        return init_ops.fill(shape, value=0, dtype=dtype)


def glorot_uniform_initializer(dtype='float32'):
    return variance_scaling_initializer(
        scale=1.0,
        mode='fan_avg',
        distribution='uniform',
        dtype=dtype,
    )


def glorot_normal_initializer(dtype='float32'):
    return variance_scaling_initializer(
        scale=1.0,
        mode='fan_avg',
        distribution='normal',
        dtype=dtype,
    )


# Aliases
zeros_initializer = Zeros
ones_initializer = Ones
constant_initializer = Constant
random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
variance_scaling_initializer = VarianceScaling
