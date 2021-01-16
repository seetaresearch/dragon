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

from dragon.core.ops import vision_ops
from dragon.vm.tensorlayer.core.engine import layer
from dragon.vm.tensorlayer.core.layers import utils
from dragon.vm.tensorlayer.core.layers.convolution import conv_utils


class Conv2d(layer.Layer):
    r"""2d convolution layer.

    Examples:

    ```python
    # Define and call a conv2d layer with the input
    x = tl.layers.Input([8, 3, 400, 400])
    y = tl.layers.Conv2d(n_filter=32, filter_size=3, stride=2)(x)
    ```

    """

    def __init__(
        self,
        n_filter,
        filter_size=3,
        strides=1,
        act=None,
        padding='SAME',
        data_format='channels_first',
        dilation_rate=1,
        W_init='glorot_uniform',
        b_init=None,
        in_channels=None,
        name=None,
    ):
        """Create a ``Conv2d`` layer.

        Parameters
        ----------
        n_filter : int, required
            The number of output filters.
        filter_size : Sequence[int], optional, default=3
            The size of convolution window.
        strides : Union[int, Sequence[int]], optional, default=1
            The stride of convolution window.
        act : callable, optional
            The optional activation function.
        padding : Union[int, Sequence[int], str], optional, default='SAME'
            The padding algorithm or size.
        data_format : str, optional, default='channels_first'
            ``'channels_first'`` or ``'channels_last'``.
        dilation_rate : Sequence[int], optional
            The rate of dilated convolution.
        W_init : Union[callable, str], optional
            The initializer for weight tensor.
        b_init : Union[callable, str], optional
            The initializer for bias tensor.
        in_channels : int, optional
            The number of input channels.
        name : str, optional
            The layer name.

        """
        super().__init__(name, act)
        self.n_filter = n_filter
        self.filter_size = conv_utils.normalize_2d_args('ksize', filter_size)
        self.strides = conv_utils.normalize_2d_args('strides', strides)
        self.dilation_rate = conv_utils.normalize_2d_args('dilations', dilation_rate)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.padding = padding
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels
        self.W = None
        self.b = None
        if self.in_channels:
            self.build(None)
            self._built = True

    def __repr__(self):
        s = ('{classname}('
             'in_channels={in_channels}, '
             'out_channels={n_filter}, '
             'kernel_size={filter_size}, '
             'strides={strides}, '
             'padding={padding}')
        if self.dilation_rate != (1, ) * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + utils.get_act_str(self.act))
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_channels is None:
            if self.data_format == 'channels_last':
                self.in_channels = inputs_shape[-1]
            else:
                self.in_channels = inputs_shape[1]
        filter_shape = [self.n_filter] + list(self.filter_size)
        if self.data_format == 'channels_first':
            filter_shape.insert(1, self.in_channels)
        else:
            filter_shape.append(self.in_channels)
        self.W = self.add_weight(
            name='filters',
            shape=filter_shape,
            init=self.W_init,
        )
        if self.b_init:
            self.b = self.add_weight(
                name='biases',
                shape=(self.n_filter,),
                init=self.b_init,
            )

    def forward(self, inputs, **kwargs):
        data_format = conv_utils.convert_data_format(self.data_format)
        padding, pads = conv_utils.normalize_2d_args('padding', self.padding)
        outputs = vision_ops.conv2d(
            [inputs, self.W] + ([self.b] if self.b_init else []),
            kernel_shape=self.filter_size,
            strides=self.strides,
            pads=pads,
            padding=padding,
            dilations=self.dilation_rate,
            data_format=data_format,
        )
        if self.act:
            outputs = self.act(outputs)
        return outputs
