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

from dragon.vm.tensorlayer.core.engine import layer
from dragon.vm.tensorlayer.core import activations


class Relu(layer.Layer):
    r"""The layer to apply the rectified linear unit.
    `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    The **ReLU** function is defined as:

    .. math::
        \text{ReLU}(x) =
            \begin{cases}
                \min(x, v_{max}), & \text{ if } x \geq 0 \\
                \alpha * x, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    x = tl.layers.Input([10, 5])
    y = tl.layers.Relu(channel_shared=True)
    ```

    """

    def __init__(self, inplace=False, name=None):
        """Create a ``Relu`` layer.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.
        name : str, optional
            The optional layer name.

        """
        super(Relu, self).__init__(name)
        self.inplace = inplace

    def forward(self, inputs):
        return activations.relu(inputs, inplace=self.inplace)

    def __repr__(self):
        s = '{classname}('
        s += 'inplace={inplace},'
        s += 'name={name}'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)
