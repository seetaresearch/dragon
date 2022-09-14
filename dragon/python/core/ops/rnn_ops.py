# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""RNN ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

from dragon.core.autograph import context
from dragon.core.framework.tensor import Tensor
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.op_lib import OpSchema
from dragon.core.ops import random_ops
from dragon.core.util import math_util
from dragon.core.util import nest


class RNNModule(object):
    """A simple class wrapping general RNN operations."""

    def __init__(
        self,
        mode,
        input_size,
        hidden_size,
        num_layers=1,
        bidirectional=False,
        dropout=0,
        **kwargs
    ):
        self._mode = mode.lower()
        self._num_gates = {'lstm': 4, 'gru': 3}.get(self._mode, 1)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = float(dropout) if dropout > 0 else None
        self._bidirectional = bidirectional
        self._num_directions = 2 if bidirectional else 1
        self._weights_shapes = []
        self._create_weights()
        self._initialize_weights()

    @property
    def weights(self):
        """Return the flatten RNN weights."""
        return self._weights

    def call(self, inputs, training=False, **kwargs):
        """Compute the output of RNN."""
        inputs = nest.flatten(inputs)
        inputs.insert(1, self._weights)
        if context.executing_eagerly():
            return OpLib.execute(
                'Recurrent', inputs, rnn_mode=self._mode,
                num_layers=self._num_layers, hidden_size=self._hidden_size,
                bidirectional=self._bidirectional, dropout=self._dropout,
                phase='TRAIN' if training else 'TEST')
        return OpLib.add(
            'Recurrent', inputs, rnn_mode=self._mode,
            num_layers=self._num_layers, hidden_size=self._hidden_size,
            bidirectional=self._bidirectional, dropout_ratio=self._dropout, **kwargs)

    def _create_weights(self):
        """Create a flat weights."""
        gate_size = self._hidden_size * self._num_gates
        # Compute the shape of weight and bias.
        matrix_shapes, bias_shapes = [], []
        for layer in range(self._num_layers):
            for direction in range(self._num_directions):
                layer_input_size = self._input_size if layer == 0 \
                    else self._hidden_size * self._num_directions
                w_ih_shape = [gate_size, layer_input_size]
                w_hh_shape = [gate_size, self._hidden_size]
                b_ih_shape, b_hh_shape = [gate_size], [gate_size]
                matrix_shapes.extend([w_ih_shape, w_hh_shape])
                bias_shapes.extend([b_ih_shape, b_hh_shape])
        # Create single float32 weights.
        weights_count = 0
        self._weights_shapes = matrix_shapes + bias_shapes
        for shape in self._weights_shapes:
            weights_count += math_util.prod(shape)
        self._weights = Tensor([weights_count])
        self._weights.requires_grad = True

    def _initialize_weights(self):
        """Initialize the flatten weights."""
        stddev = 1.0 / math.sqrt(self._hidden_size)
        for layer_id, param_id in itertools.product(
                range(self._num_layers * (self._bidirectional + 1)),
                range(self._num_gates * 2)):
            i = layer_id * 2 + (param_id // self._num_gates)
            j = i + len(self._weights_shapes) // 2
            matrix_shape = self._weights_shapes[i][:]
            bias_shape = self._weights_shapes[j][:]
            matrix_shape[0] //= self._num_gates
            bias_shape[0] //= self._num_gates
            self._set_parameter(
                random_ops.random_uniform(matrix_shape, -stddev, stddev),
                layer_id, param_id, 'matrix')
            self._set_parameter(
                random_ops.random_uniform(bias_shape, -stddev, stddev),
                layer_id, param_id, 'bias')

    def _set_parameter(self, data, layer_id=0, param_id=0, param_type='matrix'):
        """Set the data of a parameter."""
        return OpLib.execute(
            'RNNParamSet', [data], outputs=[self._weights],
            rnn_mode=self._mode, bidirectional=self._bidirectional,
            input_size=self._input_size, hidden_size=self._hidden_size,
            layer_id=layer_id, param_id=param_id, param_type=param_type)

    def __call__(self, *args, **kwargs):
        return self.call(args, **kwargs)


class RNN(RNNModule):
    r"""Apply a multi-layer Elman RNN.
    `[Elman, 1990] <https://doi.org/10.1016>`_.

    The data format of inputs should be :math:`(T, N, C)`:

    ```python
    t, n, c = 8, 2, 4
    m = dragon.nn.RNN(8, 16)
    x = dragon.constant([t, n, c], 'float32')
    y = m(x)
    ```

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity='relu',
        num_layers=1,
        bidirectional=False,
        dropout=0,
        **kwargs
    ):
        """Create a ``RNN`` module.

        Parameters
        ----------
        input_size : int
            The dimension of input.
        hidden_size : int
            The dimension of hidden state.
        nonlinearity : {'tanh', 'relu'}, optional
            The nonlinearity.
        num_layers : int, optional, default=1
            The number of recurrent layers.
        bidirectional : bool, optional, default=False
            Whether to create a bidirectional rnn.
        dropout : number, optional, default=0
            The dropout ratio.

        """
        mode = 'rnn_relu' if nonlinearity == 'relu' else 'rnn_tanh'
        super(RNN, self).__init__(
            mode, input_size, hidden_size,
            num_layers, bidirectional, dropout, **kwargs
        )


class LSTM(RNNModule):
    r"""Apply a multi-layer long short-term memory (LSTM) RNN.
    `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    The data format of inputs should be :math:`(T, N, C)`:

    ```python
    t, n, c = 8, 2, 4
    m = dragon.nn.LSTM(8, 16)
    x = dragon.constant([t, n, c], 'float32')
    y = m(x)
    ```

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bidirectional=False,
        dropout=0,
        **kwargs
    ):
        """Create a ``LSTM`` module.

        Parameters
        ----------
        input_size : int
            The dimension of input.
        hidden_size : int
            The dimension of hidden state.
        num_layers : int, optional, default=1
            The number of recurrent layers.
        bidirectional : bool, optional, default=False
            Whether to create a bidirectional lstm.
        dropout : number, optional, default=0
            The dropout ratio.

        """
        super(LSTM, self).__init__(
            'lstm', input_size, hidden_size,
            num_layers, bidirectional, dropout, **kwargs
        )


class GRU(RNNModule):
    """Apply a multi-layer gated recurrent unit (GRU) RNN.
    `[Cho et.al, 2014] <https://arxiv.org/abs/1406.1078>`_.

    The data format of inputs should be :math:`(T, N, C)`:

    ```python
    t, n, c = 8, 2, 4
    m = dragon.nn.GRU(8, 16)
    x = dragon.constant([t, n, c], 'float32')
    y = m(x)
    ```

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bidirectional=False,
        dropout=0,
        **kwargs
    ):
        """Create a ``GRU`` module.

        Parameters
        ----------
        input_size : int
            The dimension of input.
        hidden_size : int
            The dimension of hidden state.
        num_layers : int, optional, default=1
            The number of recurrent layers.
        bidirectional : bool, optional, default=False
            Whether to create a bidirectional lstm.
        dropout : number, optional, default=0
            The dropout ratio.

        """
        super(GRU, self).__init__(
            'gru', input_size, hidden_size,
            num_layers, bidirectional, dropout, **kwargs
        )


@OpSchema.num_inputs(2)
def lstm_cell(inputs, **kwargs):
    r"""Apply a long short-term memory (LSTM) cell.
    `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The **x.4** and **cx**.

    Returns
    -------
    Sequence[dragon.Tensor]
        The input "h" and "c" tensor.

    """
    if context.executing_eagerly():
        return OpLib.execute('LSTMCell', inputs, outputs=[None, None])
    return OpLib.add('LSTMCell', num_outputs=2, **kwargs)
