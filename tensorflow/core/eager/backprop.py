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
"""Backprop engine."""

from dragon.core.autograph import backprop


class GradientTape(backprop.GradientTape):
    """Record the operations for auto differentiation.

    You should enter a tape before the execution performed:

    ```python
    with dragon.eager_mode():
        x = tf.ones(shape=(2, 3))
        with tf.GradientTape() as tape:
            y = x + 1
        print(tape.gradient(y, x))  # None, as ``x`` is not watched

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = x + 1
        print(tape.gradient(y, x))  # Ok
    ```

    """

    def __init__(self, persistent=False):
        """Create a ``GradientTape``.

        Parameters
        ----------
        persistent : bool, optional, default=False
            ``False`` release resources once ``gradient(...)`` called.

        """
        super(GradientTape, self).__init__(persistent)
