# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Random ops."""

try:
    from nvidia.dali import ops
except ImportError:
    from dragon.core.util import deprecation

    ops = deprecation.not_installed("nvidia.dali")

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
        if isinstance(dtype, str):
            dtype = getattr(types, dtype.upper())
        return ops.random.CoinFlip(probability=probability, dtype=dtype, **kwargs)


class Uniform(object):
    """Sample values from an uniform distribution.

    Examples:

    ```python
    uniform_rng = dali.ops.Uniform((0., 1.))
    value = uniform_rng()
    ```

    """

    def __new__(cls, range=(-1.0, 1.0), dtype=None, **kwargs):
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
        if isinstance(dtype, str):
            dtype = getattr(types, dtype.upper())
        return ops.random.Uniform(range=range, dtype=dtype, **kwargs)
