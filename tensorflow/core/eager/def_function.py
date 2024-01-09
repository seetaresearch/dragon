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
"""Function engine."""

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
