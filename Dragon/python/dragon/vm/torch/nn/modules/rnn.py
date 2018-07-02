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

import warnings
import numbers
import numpy as np
import dragon as dg

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.nn import Module, Parameter
from dragon.operators.rnn.rnn_param import RNNParamSet
from dragon.vm.torch.module import RunOperator
from dragon.vm.torch.autograd.grad_mode import is_grad_enabled


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

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
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
        self.meta_in_phase = {'TRAIN': [None, None], 'TEST': [None, None]}

    def register_op(self):
        self.op_meta = {
            'op_type': 'Recurrent',
            'n_inputs': 4 , 'n_outputs': 2, # meaningless
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

    def make_meta_from_phase(self, phase):
        def reset_meta(self, phase):
            # Ren-Gen Key
            self._persistent_key = None
            _ = self.persistent_key
            self._persistent_key += '/{}'.format(phase)
            self.op_meta['arguments']['phase'] = phase
            # Re-Gen Op
            self._gen_op()
            self.meta_in_phase[phase][0] = self._persistent_key
            self.meta_in_phase[phase][1] = self._op

        if self._persistent_key is None:
            # Init or CTX has changed
            reset_meta(self, phase)
        else:
            # CTX unchanged & Run into a new phase
            if self.meta_in_phase[phase][0] is None:
                reset_meta(self, phase)

        return self.meta_in_phase[phase]

    def forward(self, input, hx=None):
        if hx and not isinstance(hx, Tensor):
            raise TypeError('Excepted hx as a Tensor, got {}.'.format(type(hx)))

        if not self._init_params: self._reset_params()

        inputs = [input, self.weights] + ([hx] if hx else [])
        self.unify_devices(inputs)
        outputs = [self.register_output(input.dtype) for _ in range(2)]

        requires_grad = False
        for input in inputs:
            if input.requires_grad: requires_grad = True
        requires_grad = requires_grad and is_grad_enabled()
        meta = ['PERSISTENT',] + self.make_meta_from_phase(
            'TRAIN' if requires_grad else 'TEST')

        return RunOperator(inputs, outputs, meta)

    def _plan_params(self):
        if self.mode == 'lstm': gate_size = 4 * self.hidden_size
        elif self.mode == 'gru': gate_size = 3 * self.hidden_size
        else: gate_size = self.hidden_size
        # 1. plan weights
        self._matrix_weights = []; self._bias_weights = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                layer_input_size = self.input_size if layer == 0 \
                    else self.hidden_size * self.num_directions
                w_names = ['layer_{}/{}/{}'.format(layer, p, 'L' if direction == 0 else 'R')
                           for p in ('matrix_ih', 'matrix_hh', 'bias_ih', 'bias_hh')]
                w_ih = dg.Tensor(name=w_names[0], shape=[gate_size, layer_input_size])
                w_hh = dg.Tensor(name=w_names[1], shape=[gate_size, self.hidden_size])
                b_ih = dg.Tensor(name=w_names[2], shape=[gate_size,])
                b_hh = dg.Tensor(name=w_names[3], shape=[gate_size,])
                # W (0 ~ 3), R (4 ~ 7)
                self._matrix_weights.extend([w_ih, w_hh])
                # Bw (0 ~ 3), Br (4 ~ 7)
                self._bias_weights.extend([b_ih, b_hh])

        # 2. compute total number of parameters
        self._weights_count = 0
        for w in self._matrix_weights + self._bias_weights:
            self._weights_count += np.prod(w.shape)

        # 3. register the packed weights
        self.weights = Parameter(Tensor(int(self._weights_count)))

        # 4. create the initialization grids
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

        # 5. set the init flag
        self._init_params = False

    ##############################################
    #                                            #
    #                INITIALIZER                 #
    #                                            #
    ##############################################

    def _uniform_init(self, shape, dtype='float32'):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        return np.random.uniform(-stdv, stdv, shape).astype(dtype)

    def _orthogonal_init(self, shape, gain=1, dtype='float32'):
        num_rows = 1
        for dim in shape[:-1]: num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_cols, num_rows) if num_rows < num_cols \
            else (num_rows,  num_cols)
        W = np.random.randn(*flat_shape)
        q, r = np.linalg.qr(W)
        # Make Q uniform
        d = np.diag(r)
        q *= np.sign(d)
        if num_rows < num_cols: q = q.T
        return gain * q.reshape(shape).astype(dtype)

    def _zero_init(self, shape, dtype='float32'):
        return np.zeros(shape, dtype=dtype)

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
        if not isinstance(param, Tensor):
            if isinstance(param, np.ndarray):
                paramT = dg.Tensor('/tmp/rnn_param').Variable()
                paramT.set_value(param)
                param = paramT
            else: raise ValueError('Excepted a tensor or numpy array.')
        W = self.weights.dragon()
        outputs = RNNParamSet([W, param], layer_id, param_id, param_type,
            rnn_mode=self.mode, input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, num_directions=self.num_directions)
        for k, v in outputs.expressions.items(): dg.workspace.RunOperator(v)

    def _reset_params(self):
        np.random.seed(dg.config.GetRandomSeed())
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
                    matrix_shape = self._matrix_weights[packed_id].shape[:]
                    bias_shape = self._bias_weights[packed_id].shape[:]
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