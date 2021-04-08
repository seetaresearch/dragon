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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/backprop.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
