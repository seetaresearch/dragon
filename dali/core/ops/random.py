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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from nvidia.dali import ops
except ImportError:
    ops = None


class CoinFlip(object):
    """Sample values from a bernoulli distribution.

    Examples:

    ```python
    flip_rng = dali.ops.CoinFlip(0.5)
    value = flip_rng()
    ```

    """

    def __new__(cls, probability=0.5):
        """Create a ``CoinFlip`` operator.

        Parameters
        ----------
        probability : float, optional, default=0.5
            The probability to return 1.

        Returns
        -------
        nvidia.dali.ops.CoinFlip
            The operator.

        """
        return ops.CoinFlip(probability=probability)


class Uniform(object):
    """Sample values from a uniform distribution.

    Examples:

    ```python
    uniform_rng = dali.ops.Uniform((0., 1.))
    value = uniform_rng()
    ```

    """

    def __new__(cls, range=(-1., 1.)):
        """Create an ``Uniform`` operator.

        Parameters
        ----------
        range : Tuple[float, float], optional
            The lower and upper bound of distribution.

        Returns
        -------
        nvidia.dali.ops.Uniform
            The operator.

        """
        return ops.Uniform(range=range)
