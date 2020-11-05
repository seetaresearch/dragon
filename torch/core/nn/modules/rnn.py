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
"""RNN modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
import numbers
import numpy

from dragon.core.framework import config
from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules import _functions as nn_funcs
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.ops.init import functional as init_funcs
from dragon.vm.torch.core.tensor import Tensor


class RNNBase(Module):
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
        """
        Initialize the module.

        Args:
            self: (todo): write your description
            mode: (todo): write your description
            input_size: (int): write your description
            hidden_size: (int): write your description
            num_layers: (int): write your description
            bias: (float): write your description
            batch_first: (str): write your description
            dropout: (str): write your description
            bidirectional: (str): write your description
        """
        super(RNNBase, self).__init__()
        self.mode = mode
        self.num_gates = {'lstm': 4, 'gru': 3}.get(self.mode, 1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout if dropout != 0 else None
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        if batch_first:
            raise NotImplementedError('Batch first is disabled.')
        if not bias:
            raise NotImplementedError('Bias is required.')
        if not isinstance(dropout, numbers.Number) or \
                not 0 <= dropout <= 1 or isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being zeroed.")
        self._register_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        numpy.random.seed(config.config().random_seed)
        for li, di, pi in itertools.product(
            range(self.num_layers),
            range(self.num_directions),
            range(self.num_gates * 2),
        ):
            self.reset_parameter(li, di, pi, 'matrix', 'orthogonal')
            self.reset_parameter(li, di, pi, 'bias', 'zero')

    def reset_parameter(
        self,
        layer_id=0,
        direction=0,
        param_id=0,
        type='matrix',
        initializer='orthogonal',
    ):
        """Reset the specific parameter.

        Parameters
        ----------
        layer_id : int, optional, default=0
            The layer id.
        direction : {0, 1}, optional, default=0
            The direction flag.
        param_id : int, optional, default=0
            The param id.
        type : {'matrix', 'bias'}, optional
            The param type.
        initializer : {'orthogonal', 'uniform', 'zero'}, optional
            The optional initializer.

        """
        li, di, pi = layer_id, direction, param_id
        li = li * self.num_directions + di
        si = li * 2 + (pi // self.num_gates)
        init_fn = getattr(self, '_{}_init'.format(initializer))
        param_shape = getattr(self, '_{}_shape'.format(type))[si][:]
        param_shape[0] //= self.num_gates  # Gate-Agnostic
        self._set_parameter(li, pi, type, init_fn(param_shape))

    def extra_repr(self):
        """
        Return a human - readable representation.

        Args:
            self: (todo): write your description
        """
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def forward(self, input, hx=None):
        """
        Perform computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            hx: (todo): write your description
        """
        return nn_funcs.Recurrent \
            .instantiate(
                self.weights.device,
                mode=self.mode,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                bidirectional=self.bidirectional,
                dropout_ratio=self.dropout,
                is_training=self.training,
            ).apply(input, self.weights, hx)

    @classmethod
    def _orthogonal_init(cls, shape, gain=1, dtype='float32'):
        """The orthogonal initializer."""
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_cols, num_rows) if num_rows < num_cols \
            else (num_rows, num_cols)
        w = numpy.random.randn(*flat_shape)
        q, r = numpy.linalg.qr(w)
        # Make Q uniform
        d = numpy.diag(r)
        q *= numpy.sign(d)
        if num_rows < num_cols:
            q = q.T
        return gain * q.reshape(shape).astype(dtype)

    def _register_parameters(self):
        """Register and flatten the parameters."""
        if self.mode == 'lstm':
            gate_size = 4 * self.hidden_size
        elif self.mode == 'gru':
            gate_size = 3 * self.hidden_size
        else:
            gate_size = self.hidden_size
        # Compute the shape of weight and bias.
        self._matrix_shape, self._bias_shape = [], []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                layer_input_size = self.input_size if layer == 0 \
                    else self.hidden_size * self.num_directions
                w_ih_shape = [gate_size, layer_input_size]
                w_hh_shape = [gate_size, self.hidden_size]
                b_ih_shape, b_hh_shape = [gate_size], [gate_size]
                # W (0 ~ 3), R (4 ~ 7)
                self._matrix_shape.extend([w_ih_shape, w_hh_shape])
                # Bw (0 ~ 3), Br (4 ~ 7)
                self._bias_shape.extend([b_ih_shape, b_hh_shape])
        # Compute total number of parameters.
        self._weights_count = 0
        for shape in self._matrix_shape + self._bias_shape:
            self._weights_count += int(numpy.prod(shape))
        # Create the flat float32 weights.
        self.weights = Parameter(Tensor(self._weights_count))

    def _set_parameter(self, layer_id, param_id, param_type, param):
        """Set parameter to the flatten weights."""
        if isinstance(param, numpy.ndarray):
            param = Tensor(
                param,
                copy=False,
                requires_grad=self.weights.requires_grad,
            )
        return nn_funcs.RNNParamSet \
            .instantiate(
                self.weights.device,
                layer_id=layer_id,
                param_id=param_id,
                param_type=param_type,
                mode=self.mode,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_directions=self.num_directions,
            ).apply(param, self.weights)

    def _uniform_init(self, shape, dtype='float32'):
        """The uniform initializer."""
        stdv = 1. / numpy.sqrt(self.hidden_size)
        return numpy.random.uniform(-stdv, stdv, shape).astype(dtype)

    @classmethod
    def _zero_init(cls, shape, dtype='float32'):
        """The zero initializer."""
        return numpy.zeros(shape, dtype=dtype)


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
        nonlinearity='relu',
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
            **True** to use bias.
        batch_first : bool, optional, default=False
            **True** to use order **[N, T, C]** otherwise **[T, N, C]**.
        dropout : number, optional, default=0
            The dropout ratio.
        bidirectional : bool, optional, default=False
            Whether to create a bidirectional rnn.

        """
        mode = 'rnn_relu' if nonlinearity == 'relu' else 'rnn_tanh'
        super(RNN, self).__init__(
            mode, input_size, hidden_size, num_layers,
            bias, batch_first, dropout, bidirectional,
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
            **True** to use bias.
        batch_first : bool, optional, default=False
            **True** to use order **[N, T, C]** otherwise **[T, N, C]**.
        dropout : number, optional, default=0
            The dropout ratio.
        bidirectional : bool, optional, default=False
            Whether to create a bidirectional lstm.

        """
        super(LSTM, self).__init__(
            'lstm', input_size, hidden_size, num_layers,
            bias, batch_first, dropout, bidirectional,
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
            **True** to use bias.
        batch_first : bool, optional, default=False
            **True** to use order **[N, T, C]** otherwise **[T, N, C]**.
        dropout : number, optional, default=0
            The dropout ratio.
        bidirectional : bool, optional, default=False
            Whether to create a bidirectional gru.

        """
        super(GRU, self).__init__(
            'gru', input_size, hidden_size, num_layers,
            bias, batch_first, dropout, bidirectional,
        )


class RNNCellBase(Module):
    def __init__(self, input_size, hidden_size, bias, num_chunks):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
            input_size: (int): write your description
            hidden_size: (int): write your description
            bias: (float): write your description
            num_chunks: (int): write your description
        """
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
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def extra_repr(self):
        """
        Return a string representation of this object.

        Args:
            self: (todo): write your description
        """
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        """
        Reset the model parameters.

        Args:
            self: (todo): write your description
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


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
            **True** to use bias.

        """
        super(LSTMCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=4)

    def forward(self, input, hx=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            hx: (todo): write your description
        """
        if hx is None:
            zeros = init_funcs.zeros(
                input.size(0),
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (zeros, zeros)
        wx = F.linear(input, self.weight_ih, self.bias_ih)
        wh = F.linear(hx[0], self.weight_hh, self.bias_hh)
        return F.lstm_cell(wx + wh, hx[1])
