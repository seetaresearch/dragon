# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.tensorlayer.core import initializers
from dragon.vm.tensorlayer.core.engine import layer


class _InputLayer(layer.Layer):
    """The starting layer of a neural network.

    Parameters
    ----------
    shape : Sequence[int]
        Including batch size.
    dtype: str, optional, default='float32'
        The data type of input values.
    name : str, optional
       The layer name.

    """

    def __init__(self, shape, dtype='float32', name=None):
        super(_InputLayer, self).__init__(name)
        self.shape, self.dtype = shape, dtype
        shape_without_none = [_ if _ is not None else 1 for _ in shape]
        outputs = initializers.ones()(shape_without_none, dtype=dtype)
        self._add_node([], outputs)

    def __repr__(self):
        s = 'Input(shape=%s' % str(self.shape)
        if self.name is not None:
            s += (', name=\'%s\'' % self.name)
        s += ')'
        return s

    def __call__(self, inputs, *args, **kwargs):
        return super(_InputLayer, self).__call__(inputs)

    def forward(self, inputs):
        return inputs


def Input(shape, dtype='float32', name=None):
    """Create a placeholder as input.

    The placeholder is an eager tensor filled with ones:

    ```python
    x = tl.layers.Input(shape=(2, 3))
    print(x)
    ```

    You can map the memory of a constant value to it:

    ```python
    value = [1, 2]
    x.set_value(value)
    x += 1
    print(x, value)  # x != value, python list could not be mapped

    value = np.array([1, 2])
    x.set_value(value)
    x += 1
    print(x, value)  # x == value, memory is zero-copied
    ```

    Parameters
    ----------
    shape : Sequence[int]
        The tensor shape to initialize values.
    dtype : str, optional, default='float32'
        The optional data type.
    name : str, optional
        The operation name.

    Returns
    -------
    dragon.Tensor
        The placeholder tensor.

    """
    input_layer = _InputLayer(shape, dtype=dtype, name=name)
    outputs = input_layer._nodes[0].out_tensors[0]
    return outputs
