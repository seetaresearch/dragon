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
"""Clip operators."""

from dragon.core.ops import math_ops


def clip_by_value(t, clip_value_min, clip_value_max, name=None):
    r"""Clip input according to the given bounds.

    .. math:: \text{out} = \min(\max(x, \text{low}), \text{high})

    Examples:

    ```python
    x = tf.constant([-2, -1, 0, 1, 2])
    print(tf.clip_by_value(x, clip_value_min=0, clip_value_max=1))
    ```

    Parameters
    ----------
    t : dragon.Tensor
        The tensor :math:`x`.
    clip_value_min : number, optional
        The value to :math:`\text{low}`.
    clip_value_max : number, optional
        The value to :math:`\text{high}`.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The output tensor.

    """
    return math_ops.clip(t, clip_value_min, clip_value_max, name=name)
