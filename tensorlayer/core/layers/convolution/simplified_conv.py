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
from dragon.vm.tensorlayer.core.layers import utils
from dragon.vm.tensorlayer.core.layers.convolution import conv_utils


class Conv2d(layer.Layer):
    r"""The 2d convolution layer.

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
            The size of filter.
        strides : Sequence[int], optional, default=1
            The stride(s) of sliding window.
        act : callable, optional
            The optional activation function.
        padding : Union[{'VALID', 'SAME'}, Sequence[int]]
            The padding algorithm or padding sizes.
        data_format : {'channels_first', 'channels_last'}, optional
             The optional data format.
        dilation_rate : Sequence[int], optional
            The rate(s) of dilated kernel.
        W_init : Union[callable, str], optional
            The initializer for weight matrix.
        b_init : Union[callable, str], optional
            The initializer for bias vector.
        in_channels : int, optional
            The number of input channels.
        name : str, optional
            The optional layer name.

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
        # Fake shape with ``channels_first`` format,
        # to indicate the backend to compute fans correctly.
        filter_shape = [self.n_filter, self.in_channels] + self.filter_size
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
