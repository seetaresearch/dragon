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
"""Math ops."""

try:
    from nvidia.dali import ops
except ImportError:
    from dragon.core.util import deprecation

    ops = deprecation.not_installed("nvidia.dali")

from dragon.vm.dali.core.framework import context
from dragon.vm.dali.core.framework import types


class Normalize(object):
    """Normalize input.

    Examples:

    ```python
    norm = dali.ops.Normalize(
        # Batch normalization case of HWC layout
        axes=(0, 1), batch=True, epsilon=1e-5,
    )
    y = norm(inputs['x'])
    ```

    """

    def __new__(
        cls,
        axes=(0, 1),
        mean=None,
        stddev=None,
        scale=1.0,
        shift=0.0,
        batch=False,
        epsilon=0,
        dtype="float32",
        **kwargs
    ):
        """Create a ``Normalize`` operator.

        Parameters
        ----------
        axes : Sequence[int], optional
            The axes to normalize.
        mean : float, optional
            The value to subtract.
        stddev : float, optional
            The value to divide after subtraction.
        scale : float, optional, default=1.0
            The scale factor after normalization.
        shift : float, optional, default=0.0
            The shift factor after normalization.
        batch : bool, optional, default=False
            Whether to compute mean and stddev across the batch.
        epsilon : float, optional, default=0
            The value added to the computed variance.
        dtype : str, optional, default='float32'
            The output data type.

        Returns
        -------
        nvidia.dali.ops.Normalize
            The operator.

        """
        if isinstance(dtype, str):
            dtype = getattr(types, dtype.upper())
        return ops.Normalize(
            axes=axes,
            mean=mean,
            stddev=stddev,
            scale=scale,
            shift=shift,
            batch=batch,
            epsilon=epsilon,
            dtype=dtype,
            device=context.get_device_type(),
            **kwargs
        )
