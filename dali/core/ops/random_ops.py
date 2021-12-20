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
"""Random ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from nvidia.dali import ops
except ImportError:
    from dragon.core.util import deprecation
    ops = deprecation.not_installed('nvidia.dali')

from dragon.core.util import six
from dragon.vm.dali.core.framework import types


class CoinFlip(object):
    """Sample values from a bernoulli distribution.

    Examples:

    ```python
    flip_rng = dali.ops.CoinFlip(0.5)
    value = flip_rng()
    ```

    """

    def __new__(cls, probability=0.5, dtype=None, **kwargs):
        """Create a ``CoinFlip`` operator.

        Parameters
        ----------
        probability : float, optional, default=0.5
            The probability to return 1.
        dtype : str, optional
            The output data type.

        Returns
        -------
        nvidia.dali.ops.random.CoinFlip
            The operator.

        """
        if isinstance(dtype, six.string_types):
            dtype = getattr(types, dtype.upper())
        return ops.random.CoinFlip(probability=probability,
                                   dtype=dtype, **kwargs)


class Uniform(object):
    """Sample values from an uniform distribution.

    Examples:

    ```python
    uniform_rng = dali.ops.Uniform((0., 1.))
    value = uniform_rng()
    ```

    """

    def __new__(cls, range=(-1., 1.), dtype=None, **kwargs):
        """Create an ``Uniform`` operator.

        Parameters
        ----------
        range : Tuple[float, float], optional
            The lower and upper bound of distribution.
        dtype : str, optional
            The output data type.

        Returns
        -------
        nvidia.dali.ops.random.Uniform
            The operator.

        """
        if isinstance(dtype, six.string_types):
            dtype = getattr(types, dtype.upper())
        return ops.random.Uniform(range=range, dtype=dtype, **kwargs)
