# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import dragon as dg
import numpy as np

from dragon.core.tensor import Tensor
from dragon.core.tensor_utils import FromShape

from .rnn_param import RNNParamSet


class RNNBase(object):
    """A simple class wrapping general RNN ops.

    """
    def __init__(self,
        mode, input_size, hidden_size, num_layers=1,
            bidirectional=False, dropout=0, name=None
    ):
        eligible_rnn_modes = ('rnn_tanh', 'rnn_relu', 'lstm', 'gru')
        if mode.lower() not in eligible_rnn_modes:
            raise ValueError('Unknown rnn mode: {}.'
                '\n<RecurrentOp> supports the following rnn modes: {{\n{}\n}}'.format(
                    mode, ',\n'.join(['    * ' + emode for emode in eligible_rnn_modes])))
        if dropout > 0 and num_layers == 1:
            warnings.warn("Add dropout to single-layer RNN is meaningless.")
        self.mode = mode.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if dropout > 0 else None
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.name = name
        self._init_params = False
        self._plan_params()

    def _plan_params(self):
        if self.mode == 'lstm': gate_size = 4 * self.hidden_size
        elif self.mode == 'gru': gate_size = 3 * self.hidden_size
        else: gate_size = self.hidden_size
        # 1. Plan weights
        self._matrix_weights = []; self._bias_weights = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                layer_input_size = self.input_size if layer == 0 \
                    else self.hidden_size * self.num_directions
                w_names = ['layer_{}/{}/{}'.format(layer, p, 'L' if direction == 0 else 'R')
                           for p in ('matrix_ih', 'matrix_hh', 'bias_ih', 'bias_hh')]
                w_ih = Tensor(name=w_names[0], shape=[gate_size, layer_input_size])
                w_hh = Tensor(name=w_names[1], shape=[gate_size, self.hidden_size])
                b_ih = Tensor(name=w_names[2], shape=[gate_size,])
                b_hh = Tensor(name=w_names[3], shape=[gate_size,])
                # W (0 ~ 3), R (4 ~ 7)
                self._matrix_weights.extend([w_ih, w_hh])
                # Bw (0 ~ 3), Br (4 ~ 7)
                self._bias_weights.extend([b_ih, b_hh])

        # 2. Compute total number of parameters
        self._weights_count = 0
        for w in self._matrix_weights + self._bias_weights:
            self._weights_count += np.prod(w.shape)

        # 3. Register the packed weights
        self.weights = FromShape(shape=[self._weights_count],
            name=self.name + '/weights' if self.name else None)

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
                paramT = Tensor('/tmp/rnn_param').Variable()
                paramT.set_value(param)
                param = paramT
            else: raise ValueError('Excepted a tensor or numpy array.')
        self.weights.expressions = dict() # Clear cached expressions
        outputs = RNNParamSet([self.weights, param], layer_id, param_id, param_type,
            rnn_mode=self.mode, input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, num_directions=self.num_directions)
        for k, v in outputs.expressions.items(): dg.workspace.RunOperator(v)

    def _reset_params(self):
        np.random.seed(dg.config.GetRandomSeed())
        if self.mode == 'lstm': num_gates = 4
        elif self.mode == 'gru': num_gates = 3
        else: num_gates = 1
        weights_states = self.weights.expressions.copy()
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
        self.weights.expressions = weights_states
        self._init_params = True

    def create(self, x, hx=None, cx=None,
            required_hidden=True, required_cell=False):
        """Return outputs of this rnn.

        Parameters
        ----------
        x : Tensor
            The input tensor.
        hx : Tensor or None
            The h(0) state.
        cx : Tensor or None
            The c(0) state.
        required_hidden : boolean
            Return ``y`` and ``hidden`` if ``True``.
        required_hidden : boolean
            Return ``y``, ``hidden``, ``cell`` if ``True``.

        """
        if hx and not isinstance(hx, Tensor):
            raise TypeError('Excepted hx as a Tensor, got {}.'.format(type(hx)))
        if cx and not isinstance(cx, Tensor):
            raise TypeError('Excepted cx as a Tensor, got {}.'.format(type(cx)))

        if not self._init_params: self._reset_params()

        arguments = {
            'inputs': [x, self.weights] +
                          ([hx] if hx else []) +
                              ([cx] if cx else []),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'rnn_mode': self.mode,
            'rnn_input_mode': 'linear',
            'dropout_ratio': self.dropout,
        }

        if required_cell: n_out = 3
        elif required_hidden: n_out = 2
        else: n_out = 1

        return Tensor.CreateOperator(nout=n_out, op_type='Recurrent', **arguments)

    def __call__(self, *args, **kwargs):
        return self.create(*args, **kwargs)