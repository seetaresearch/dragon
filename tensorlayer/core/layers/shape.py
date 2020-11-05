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

from dragon.core.ops import array_ops
from dragon.vm.tensorlayer.core.engine import layer
from dragon.vm.tensorlayer.core.layers import utils


class Flatten(layer.Layer):
    """The layer to reshape input into a matrix.

    Examples:

    ```python
    x = tl.layers.Input([8, 4, 3])
    y = tl.layers.Flatten()(x)  # [8, 4, 3] -> [8, 12]
    ```

    """

    def __init__(self, name=None):
        """Create a ``Flatten`` layer.

        Parameters
        ----------
        name : str, optional
            The optional layer name.

        """
        super(Flatten, self).__init__(name)

    def __repr__(self):
        """
        Return a repr representation of this object.

        Args:
            self: (todo): write your description
        """
        s = '{classname}('
        s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, **kwargs):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return utils.flatten_reshape(inputs, name=self.name)


class Reshape(layer.Layer):
    """The layer to change the dimensions of input.

    Examples:

    ```python
    x = tl.layers.Input([8, 4, 3])
    y = tl.layers.Reshape(shape=[-1, 12])(x)  # [8, 4, 3] -> [8, 12]
    ```

    """

    def __init__(self, shape, name=None):
        """Create a ``Reshape`` layer.

        Parameters
        ----------
        shape : Sequence[int]
            The output shape.
        name : str, optional
            The optional layer name.

        """
        super(Reshape, self).__init__(name)
        self.shape = shape

    def __repr__(self):
        """
        Return a repr representation of this object.

        Args:
            self: (todo): write your description
        """
        s = '{classname}('
        s += 'shape={shape},'
        s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, **kwargs):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return array_ops.reshape(inputs, shape=self.shape)


class Transpose(layer.Layer):
    """The layer to permute the dimensions of input.

    Examples:

    ```python
    x = tl.layers.Input([8, 4, 3])

    # Specify a explict permutation
    y = tl.layers.Transpose(perm=[0, 2, 1])(x)  # [8, 4, 3] -> [8, 3, 4]

    # Or simply inverse all the dimensions
    z = tl.layers.Transpose()(x)  # [8, 4, 3] -> [3, 4, 8]
    ```

    """

    def __init__(self, perm=None, name=None):
        """Create a ``Transpose`` layer.

        Parameters
        ----------
        perm: Sequence[int], optional
            The permutation of new dimensions.
        name : str, optional
            The optional layer name.

        """
        super(Transpose, self).__init__(name)
        self.perm = perm

    def __repr__(self):
        """
        Return a repr representation of this object.

        Args:
            self: (todo): write your description
        """
        s = '{classname}('
        s += 'perm={perm},'
        s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, **kwargs):
        """
        R forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        return array_ops.transpose(inputs, perm=self.perm)
