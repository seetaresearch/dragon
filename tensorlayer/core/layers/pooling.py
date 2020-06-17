# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import vision_ops
from dragon.vm.tensorlayer.core.engine import layer
from dragon.vm.tensorlayer.core.layers.convolution import conv_utils


class MaxPool2d(layer.Layer):
    """The max 2d pooling layer.

    Examples:

    ```python
    x = tl.layers.Input([None, 32, 50, 50])
    y = tl.layers.MaxPool2d(filter_size=3, strides=2)(x)
    ```

    """

    def __init__(
        self,
        filter_size=3,
        strides=2,
        padding='SAME',
        data_format='channels_first',
        name=None,
    ):
        """Create a ``MaxPool2d`` layer.

        Parameters
        -----------
        filter_size : Sequence[int], optional, default=3
            The size of sliding window.
        strides : Sequence[int], optional, default=2
            The stride of sliding window.
        padding : Union[{'VALID', 'SAME'}, Sequence[int]]
            The padding algorithm or padding sizes.
        data_format : {'channels_first', 'channels_last'}, optional
             The optional data format.
        name : str, optional
            The optional layer name.

        """
        super(MaxPool2d, self).__init__(name)
        strides = filter_size if strides is None else strides
        self.filter_size = conv_utils.normalize_2d_args('ksize', filter_size)
        self.strides = conv_utils.normalize_2d_args('strides', strides)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.padding = padding

    def __repr__(self):
        s = '{classname}(' \
            'filter_size={filter_size}, ' \
            'strides={strides}, ' \
            'padding={padding}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, **kwargs):
        data_format = conv_utils.convert_data_format(self.data_format)
        padding, pads = conv_utils.normalize_2d_args('padding', self.padding)

        return vision_ops.pool2d(
            inputs,
            kernel_shape=self.filter_size,
            strides=self.strides,
            pads=pads,
            padding=padding,
            mode='MAX',
            global_pooling=False,
            data_format=data_format,
        )


class MeanPool2d(layer.Layer):
    """The mean 2d pooling layer.

    Examples:

    ```python
    x = tl.layers.Input([None, 32, 50, 50])
    y = tl.layers.MeanPool2d(filter_size=3, strides=2)(x)
    ```

    Parameters
    -----------
    filter_size : Sequence[int]
        The size of sliding window.
    strides : Sequence[int]
        The stride of sliding window.
    padding : Union[{'VALID', 'SAME'}, Sequence[int]]
        The padding algorithm or padding sizes.
    data_format : {'channels_last', 'channels_first'}, optional
        The optional data format.
    name : str, optional
        The optional name.

    """

    def __init__(
        self,
        filter_size=3,
        strides=2,
        padding='SAME',
        data_format='channels_first',
        name=None,
    ):
        super(MeanPool2d, self).__init__(name)
        strides = filter_size if strides is None else strides
        self.filter_size = conv_utils.normalize_2d_args('ksize', filter_size)
        self.strides = conv_utils.normalize_2d_args('strides', strides)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.padding = padding

    def __repr__(self):
        s = '{classname}(' \
            'filter_size={filter_size}, ' \
            'strides={strides}, ' \
            'padding={padding}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, **kwargs):
        data_format = conv_utils.convert_data_format(self.data_format)
        padding, pads = conv_utils.normalize_2d_args('padding', self.padding)

        return vision_ops.pool2d(
            inputs,
            kernel_shape=self.filter_size,
            strides=self.strides,
            pads=pads,
            padding=padding,
            mode='AVG',
            global_pooling=False,
            data_format=data_format,
        )


class GlobalMaxPool2d(layer.Layer):
    """The global max 2d pooling layer.

    Examples:

    ```python
    x = tl.layers.Input([None, 30, 100, 100])
    y = tl.layers.GlobalMaxPool2d()(x)
    ```

    """

    def __init__(self, data_format='channels_first', name=None):
        """Create a ``GlobalMaxPool2d`` layer.

        Parameters
        ------------
        data_format : {'channels_last', 'channels_first'}
            The optional data format.
        name : str, optional
            The optional layer name.

        """
        super(GlobalMaxPool2d, self).__init__(name)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, **kwargs):
        return vision_ops.pool2d(
            inputs,
            kernel_shape=1,
            strides=1,
            pads=0,
            mode='MAX',
            global_pooling=True,
            data_format=conv_utils.convert_data_format(self.data_format),
        )


class GlobalMeanPool2d(layer.Layer):
    """The global mean 2d pooling layer.

    Examples:

    ```python
    x = tl.layers.Input([None, 30, 100, 100])
    y = tl.layers.GlobalMeanPool2d()(x)
    ```

    Parameters
    ------------
    data_format : {'channels_last', 'channels_first'}, optional
         The optional data format.
    name : str, optional
        The optional layer name.

    """

    def __init__(self, data_format='channels_first', name=None):
        super(GlobalMeanPool2d, self).__init__(name)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, **kwargs):
        return vision_ops.pool2d(
            inputs,
            kernel_shape=1,
            strides=1,
            pads=0,
            mode='AVG',
            global_pooling=True,
            data_format=conv_utils.convert_data_format(self.data_format),
        )
