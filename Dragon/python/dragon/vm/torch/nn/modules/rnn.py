# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import warnings
import numbers
import numpy
import dragon

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.nn import Module, Parameter
from dragon.operators.rnn.rnn_param import RNNParamSet
from dragon.vm.torch.module import RunOperator
from dragon.vm.torch.ops.builtin import zeros as Zeros, xw_plus_b


class RNNBase(Module):
    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super(RNNBase, self).__init__()
        self.mode = mode
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
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))
        self._plan_params()
        self.register_op()
        self.op_metas = {'TRAIN': None, 'TEST': None}

    def register_op(self):
        self.op_meta = {
            'op_type': 'Recurrent',
            'arguments': {
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'bidirectional': self.bidirectional,
                'rnn_mode': self.mode,
                'rnn_input_mode': 'linear',
                'dropout_ratio': self.dropout,
                'phase': 'TEST',
            }
        }

    def extra_repr(self):
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

    def make_meta_from_phase(self, phase):
        def reset_meta(self, phase):
            self._module_key = None
            _ = self.module_key
            self._module_key += '/{}'.format(phase)
            self.op_meta['arguments']['phase'] = phase
            self._gen_module_def()
            self.op_metas[phase] = (self._module_key, self._module_def)

        if self._module_key is None:
            # Init or Context has changed
            reset_meta(self, phase)
        else:
            # Context unchanged
            if self.op_metas[phase] is None:
                reset_meta(self, phase)

        return self.op_metas[phase]

    def forward(self, input, hx=None):
        if hx and not isinstance(hx, Tensor):
            raise TypeError('Excepted hx as a Tensor, got {}.'.format(type(hx)))

        if not self._init_params: self._reset_params()

        inputs = [input, self.weights] + ([hx] if hx else [])
        self.unify_devices(inputs)
        outputs = [self.register_output() for _ in range(2)]

        meta = self.make_meta_from_phase(
            'TRAIN' if self.training else 'TEST')
        return RunOperator(inputs, outputs, meta)

    def _plan_params(self):
        if self.mode == 'lstm': gate_size = 4 * self.hidden_size
        elif self.mode == 'gru': gate_size = 3 * self.hidden_size
        else: gate_size = self.hidden_size
        # 1. Plan weights
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

        # 2. Compute total number of parameters
        self._weights_count = 0
        for shape in self._matrix_shape + self._bias_shape:
            self._weights_count += numpy.prod(shape)

        # 3. Register the packed weights
        self.weights = Parameter(Tensor(int(self._weights_count)))

        # 4. Create the initialization grids
        if self.mode == 'lstm': num_params_per_layer = 8
        elif self.mode == 'gru': num_params_per_layer = 6
        else: num_params_per_layer = 2
        self._matrix_init_grids = [
            [['orthogonal' for _ in range(num_params_per_layer)]
                        for _ in range(self.num_directions)]
                    for _ in range(self.num_layers)
        ]
        self._bias_init_grids = [
            [['zero' for _ in range(num_params_per_layer)]
                for _ in range(self.num_directions)]
            for _ in range(self.num_layers)
        ]

        # 5. Set the init flag
        self._init_params = False

    ##############################################
    #                                            #
    #                INITIALIZER                 #
    #                                            #
    ##############################################

    def _uniform_init(self, shape, dtype='float32'):
        stdv = 1.0 / numpy.sqrt(self.hidden_size)
        return numpy.random.uniform(-stdv, stdv, shape).astype(dtype)

    def _orthogonal_init(self, shape, gain=1, dtype='float32'):
        num_rows = 1
        for dim in shape[:-1]: num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_cols, num_rows) if num_rows < num_cols \
            else (num_rows,  num_cols)
        W = numpy.random.randn(*flat_shape)
        q, r = numpy.linalg.qr(W)
        # Make Q uniform
        d = numpy.diag(r)
        q *= numpy.sign(d)
        if num_rows < num_cols: q = q.T
        return gain * q.reshape(shape).astype(dtype)

    def _zero_init(self, shape, dtype='float32'):
        return numpy.zeros(shape, dtype=dtype)

    ##############################################
    #                                            #
    #                 PARAMETERS                 #
    #                                            #
    ##############################################

    def set_param(self, layer=0, direction=0, param_id=0,
                  type='matrix', initializer=None):
        if type == 'matrix':
            self._matrix_init_grids[layer][direction][param_id] = initializer
        elif type == 'bias':
            self._bias_init_grids[layer][direction][param_id] = initializer
        else:
            raise ValueError('Unknown param type: ' + type)

    def _set_param(self, layer_id, param_id, param_type, param):
        if isinstance(param, numpy.ndarray):
            param_temp = dragon.Tensor.Ref('/tmp/rnn_param')
            param_temp.set_value(param)
            param = param_temp
        else: raise ValueError('Excepted a numpy array.')
        W = self.weights.dragon()
        outputs = RNNParamSet([W, param], layer_id, param_id, param_type,
            rnn_mode=self.mode, input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, num_directions=self.num_directions)
        for k, v in outputs.expressions.items(): dragon.workspace.RunOperator(v)

    def _reset_params(self):
        numpy.random.seed(dragon.config.GetRandomSeed())
        if self.mode == 'lstm': num_gates = 4
        elif self.mode == 'gru': num_gates = 3
        else: num_gates = 1
        for layer in range(len(self._matrix_init_grids)):
            for direction in range(len(self._matrix_init_grids[0])):
                for param_id in range(len(self._matrix_init_grids[0][0])):
                    matrix_init = self._matrix_init_grids[layer][direction][param_id]
                    bias_init = self._bias_init_grids[layer][direction][param_id]
                    if isinstance(matrix_init, str):
                        matrix_init = getattr(self, '_{}_init'.format(matrix_init))
                    if isinstance(bias_init, str):
                        bias_init = getattr(self, '_{}_init'.format(bias_init))
                    pseudo_layer_id = layer * self.num_directions + direction
                    packed_id = pseudo_layer_id * 2 + int(param_id / num_gates)
                    matrix_shape = self._matrix_shape[packed_id][:]
                    bias_shape = self._bias_shape[packed_id][:]
                    matrix_shape[0] = bias_shape[0] = int(matrix_shape[0] / num_gates)
                    self._set_param(layer_id=pseudo_layer_id, param_id=param_id,
                        param_type='matrix', param=matrix_init(matrix_shape))
                    self._set_param(layer_id=pseudo_layer_id, param_id=param_id,
                        param_type='bias', param=bias_init(bias_shape))
        self._init_params = True


class RNN(RNNBase):
    """Multi-layer Elman-RNN with `TanH` or `ReLU` non-linearity. `[Elman, 1990] <https://doi.org/10.1016>`_.

    The data format of inputs should be ``[T, N, C]``.

    Examples
    --------
    >>> rnn = RNN(32, 64, num_layers=1, bidirectional=True)
    >>> x = torch.ones(8, 32, 256)
    >>> outputs, hidden = rnn(x)

    """
    def __init__(self, input_size, hidden_size, nonlinearity='relu',
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        """Construct a RNN module.

        Parameters
        ----------
        input_size : int
            The dimension of inputs.
        hidden_size : int
            The dimension of hidden/outputs.
        nonlinearity : str
            The nonlinearity. ``tanh`` or ``relu``.
        num_layers : int
            The number of recurrent layers.
        bias : boolean
            Whether to use bias.
        batch_first : boolean
            Whether to use order ``[N, T, C]``.
        dropout : number
            The dropout ratio. ``0`` means ``Disabled``.
        bidirectional : boolean
            Whether to use bidirectional rnn.

        Returns
        -------
        RNNBase
            The generic RNN module.

        """
        mode = 'rnn_relu' if nonlinearity == 'relu' else 'rnn_tanh'
        super(RNN, self).__init__(mode, input_size, hidden_size,
            num_layers, bias, batch_first, dropout, bidirectional)


class LSTM(RNNBase):
    """Multi-layer Long Short-Term Memory(LSTM) RNN. `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    The data format of inputs should be ``[T, N, C]``.

    Examples
    --------
    >>> rnn = LSTM(32, 64, num_layers=2, bidirectional=True, dropout=0.5)
    >>> x = torch.ones(8, 32, 256)
    >>> outputs, hidden = rnn(x)

    """
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        """Construct a LSTM module.

        Parameters
        ----------
        input_size : int
            The dimension of inputs.
        hidden_size : int
            The dimension of hidden/outputs.
        num_layers : int
            The number of recurrent layers.
        bias : boolean
            Whether to use bias.
        batch_first : boolean
            Whether to use order ``[N, T, C]``.
        dropout : number
            The dropout ratio. ``0`` means ``Disabled``.
        bidirectional : boolean
            Whether to use bidirectional rnn.

        Returns
        -------
        RNNBase
            The generic RNN module.

        """
        super(LSTM, self).__init__('lstm', input_size, hidden_size,
            num_layers, bias, batch_first, dropout, bidirectional)


class GRU(RNNBase):
    """Multi-layer Gated Recurrent Unit (GRU) RNN. `[Cho et.al, 2014] <https://arxiv.org/abs/1406.1078>`_.

    The data format of inputs should be ``[T, N, C]``.

    Examples
    --------
    >>> rnn = GRU(32, 64, num_layers=2, bidirectional=False)
    >>> x = torch.ones(8, 32, 256)
    >>> outputs, hidden = rnn(x)

    """
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        """Construct a GRU module.

        Parameters
        ----------
        input_size : int
            The dimension of inputs.
        hidden_size : int
            The dimension of hidden/outputs.
        num_layers : int
            The number of recurrent layers.
        bias : boolean
            Whether to use bias.
        batch_first : boolean
            Whether to use order ``[N, T, C]``.
        dropout : number
            The dropout ratio. ``0`` means ``Disabled``.
        bidirectional : boolean
            Whether to use bidirectional rnn.

        Returns
        -------
        RNNBase
            The generic RNN module.

        """
        super(GRU, self).__init__('gru', input_size, hidden_size,
            num_layers, bias, batch_first, dropout, bidirectional)


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
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

from .activation import Tanh, Sigmoid

class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=4)
        self.register_op()
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def register_op(self):
        self.op_meta = {'op_type': 'LSTMCell', 'arguments': {}}

    def forward(self, input, hx=None):
        if hx is None:
            zeros = Zeros(
                input.size(0), self.hidden_size,
                    dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        wx = xw_plus_b(input, self.weight_ih, self.bias_ih)
        wh = xw_plus_b(hx[0], self.weight_hh, self.bias_hh)
        inputs = [wx + wh, hx[1]]
        self.unify_devices(inputs)
        outputs = [self.register_output() for _ in range(2)]
        return self.run(inputs, outputs)