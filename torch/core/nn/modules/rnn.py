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
"""RNN modules."""

import itertools
import math

from dragon.core.util import nest
from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.nn import functional
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.ops import constant_ops
from dragon.vm.torch.core.ops import random_ops
from dragon.vm.torch.core.tensor import Tensor


class RNNBase(Module):
    """RNN base module."""

    def __init__(
        self,
        mode,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout if dropout != 0 else None
        self.bidirectional = bidirectional
        if batch_first:
            raise NotImplementedError("Batch first is not supported.")
        if not bias:
            raise NotImplementedError("Bias is required.")
        self._num_gates = {"LSTM": 4, "GRU": 3}.get(self.mode, 1)
        self._weight_count = 0
        self._weight_shapes = []
        self.weight = self.flatten_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        stddev = 1.0 / self.hidden_size**0.5
        for layer_id, param_id in itertools.product(
            range(self.num_layers * (self.bidirectional + 1)),
            range(self._num_gates * 2),
        ):
            i = layer_id * 2 + (param_id // self._num_gates)
            j = i + len(self._weight_shapes) // 2
            matrix_shape = self._weight_shapes[i][:]
            bias_shape = self._weight_shapes[j][:]
            matrix_shape[0] //= self._num_gates
            bias_shape[0] //= self._num_gates
            self._set_parameter(
                random_ops.uniform(-stddev, stddev, matrix_shape),
                layer_id,
                param_id,
                "matrix",
            )
            self._set_parameter(
                random_ops.uniform(-stddev, stddev, bias_shape),
                layer_id,
                param_id,
                "bias",
            )

    def flatten_parameters(self):
        """Flatten parameters into a single weights."""
        gate_size = self._num_gates * self.hidden_size
        matrix_shapes, bias_shapes = [], []
        for layer in range(self.num_layers):
            for direction in range(int(self.bidirectional) + 1):
                layer_input_size = (
                    self.input_size if layer == 0 else self.hidden_size * self.num_directions
                )
                w_ih_shape = [gate_size, layer_input_size]
                w_hh_shape = [gate_size, self.hidden_size]
                b_ih_shape, b_hh_shape = [gate_size], [gate_size]
                matrix_shapes.extend([w_ih_shape, w_hh_shape])
                bias_shapes.extend([b_ih_shape, b_hh_shape])
        self._weight_count = 0
        self._weight_shapes = matrix_shapes + bias_shapes
        for shape in self._weight_shapes:
            self._weight_count += math.prod(shape)
        self.weight = Parameter(Tensor(self._weight_count))
        return self.weight

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.bias is not True:
            s += ", bias={bias}"
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    def forward(self, input, hx=None):
        inputs = [input, self.weight]
        if hx is not None:
            inputs += nest.flatten(hx)
        outputs = [None] * (3 if self.mode == "LSTM" else 2)
        outputs = Function.apply(
            "RNN",
            input.device,
            inputs,
            outputs=outputs,
            rnn_mode=self.mode,
            bidirectional=self.bidirectional,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            phase="TRAIN" if self.training else "TEST",
        )
        output, hidden = outputs[0], outputs[1:]
        return output, hidden[0] if len(hidden) == 1 else hidden

    def _set_parameter(self, data, layer_id=0, param_id=0, param_type="matrix"):
        """Set the data of a parameter."""
        return Function.apply(
            "RNNParamSet",
            data.device,
            [data],
            outputs=[self.weight],
            rnn_mode=self.mode,
            bidirectional=self.bidirectional,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            layer_id=layer_id,
            param_id=param_id,
            param_type=param_type,
        )


class RNN(RNNBase):
    r"""Apply a multi-layer Elman RNN.
    `[Elman, 1990] <https://doi.org/10.1016>`_.

    Examples:

    ```python
    m = torch.nn.RNN(32, 64)
    x = torch.ones(8, 32, 256)
    outputs, hidden = m(x)
    ```

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity="relu",
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
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
        bias : bool, optional, default=True
            ``True`` to use bias.
        batch_first : bool, optional, default=False
            ``True`` to use order **[N, T, C]** otherwise **[T, N, C]**.
        dropout : number, optional, default=0
            The dropout ratio.
        bidirectional : bool, optional, default=False
            Whether to create a bidirectional rnn.

        """
        mode = "RNN_RELU" if nonlinearity == "relu" else "RNN_TANH"
        super(RNN, self).__init__(
            mode,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )


class LSTM(RNNBase):
    r"""Apply a multi-layer long short-term memory (LSTM) RNN.
    `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    Examples:

    ```python
    m = torch.nn.LSTM(32, 64)
    x = torch.ones(8, 32, 256)
    outputs, hidden = m(x)
    ```

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        """Create a ``LSTM`` module.

        input_size : int
            The dimension of input.
        hidden_size : int
            The dimension of hidden state.
        num_layers : int, optional, default=1
            The number of recurrent layers.
        bias : bool, optional, default=True
            ``True`` to use bias.
        batch_first : bool, optional, default=False
            ``True`` to use order **[N, T, C]** otherwise **[T, N, C]**.
        dropout : number, optional, default=0
            The dropout ratio.
        bidirectional : bool, optional, default=False
            Whether to create a bidirectional lstm.

        """
        super(LSTM, self).__init__(
            "LSTM",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )


class GRU(RNNBase):
    """Apply a multi-layer gated recurrent unit (GRU) RNN.
    `[Cho et.al, 2014] <https://arxiv.org/abs/1406.1078>`_.

    Examples:

    ```python
    m = torch.nn.GRU(32, 64)
    x = torch.ones(8, 32, 256)
    outputs, hidden = m(x)
    ```

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
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
        bias : bool, optional, default=True
            ``True`` to use bias.
        batch_first : bool, optional, default=False
            ``True`` to use order **[N, T, C]** otherwise **[T, N, C]**.
        dropout : number, optional, default=0
            The dropout ratio.
        bidirectional : bool, optional, default=False
            Whether to create a bidirectional gru.

        """
        super(GRU, self).__init__(
            "GRU",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )


class RNNCellBase(Module):
    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = Parameter(Tensor(num_chunks * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(Tensor(num_chunks * hidden_size))
            self.bias_hh = Parameter(Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def reset_parameters(self):
        stddev = 1.0 / self.hidden_size**0.5
        for weight in self.parameters():
            weight.data.uniform_(-stddev, stddev)


class LSTMCell(RNNCellBase):
    r"""Apply a long short-term memory (LSTM) cell.
    `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    Examples:

    ```python
    m = torch.nn.LSTMCell(32, 64)
    x = torch.ones(32, 256)
    h1, c1 = m(x)
    ```

    """

    def __init__(self, input_size, hidden_size, bias=True):
        """Create a ``LSTMCell`` module.

        Parameters
        ----------
        input_size : int
            The dimension of input.
        hidden_size : int
            The dimension of hidden state.
        bias : bool, optional, default=True
            ``True`` to use bias.

        """
        super(LSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, input, hx=None):
        if hx is None:
            zeros = constant_ops.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        wx = functional.linear(input, self.weight_ih, self.bias_ih)
        wh = functional.linear(hx[0], self.weight_hh, self.bias_hh)
        return functional.lstm_cell(wx + wh, hx[1])
