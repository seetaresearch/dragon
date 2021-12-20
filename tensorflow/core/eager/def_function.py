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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/def_function.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.autograph import function_lib


def function(func=None, input_signature=None):
    """Create a callable graph from the python function.

    Tensor operations could be compiled into graph:

    ```python
    def foo(x, y):
        return tf.add(x + y, x)

    bar = tf.function(foo)
    print(bar(1, 2))
    print(bar(tf.constant([1, 2]), tf.constant([2, 3])))
    ```

    Above usages which can simplified:

    ```python
    @tf.function
    def foo(x, y):
        return tf.add(x + y, x)

    print(foo(1, 2))
    print(foo(tf.constant([1, 2]), tf.constant([2, 3])))
    ```

    Some advanced layers require the tensor shape before compiling:

    ```python
    @tf.function
    def foo(x):
        return tf.keras.layers.Conv2D(5, 3)(x)

    print(foo(tf.constant(np.ones((1, 4, 4, 2)))))  # Missing shape

    @tf.function(input_signature=[tf.TensorSpec([None, 4, 4, 2])])
    def bar(x):
        return tf.keras.layers.Conv2D(5, 3)(x)

    print(bar(tf.constant(np.ones((1, 4, 4, 2)))))  # Ok
    ```

    Parameters
    ----------
    func : callable, optional
        The builtin python function.
    input_signature : Sequence[dragon.vm.tensorflow.TensorSpec], optional
        The indicators to the inputs.

    Returns
    -------
    callable
        The function to run the graph once.

    """
    return function_lib.function(func, input_signature)
