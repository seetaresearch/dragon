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

import numpy
import warnings

from dragon import config as _cfg
from dragon.core import workspace as _workspace
from dragon.core.tensor import Tensor as _Tensor
from dragon.core import tensor_utils as _tensor_utils

from .rnn_param import RNNParamSet


class RNNBase(object):
    """A simple class wrapping general RNN ops."""

    def __init__(self,
        mode,
        input_size,
        hidden_size,
        num_layers=1,
        bidirectional=False,
        dropout=0,
        name=None,
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
        # 1) Plan weights
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

        # 2) Compute total number of parameters
        self._weights_count = 0
        for shape in self._matrix_shape + self._bias_shape:
            self._weights_count += numpy.prod(shape)

        # 3) Register the packed weights
        self.weights = _tensor_utils.FromShape(
            shape=[self._weights_count],
            name=self.name + '/weights' if self.name else None,
        )

        # 4) Create the initialization grids
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

    def set_param(
        self,
        layer=0,
        direction=0,
        param_id=0,
        type='matrix',
        initializer=None,
    ):
        if type == 'matrix':
            self._matrix_init_grids[layer][direction][param_id] = initializer
        elif type == 'bias':
            self._bias_init_grids[layer][direction][param_id] = initializer
        else:
            raise ValueError('Unknown param type: ' + type)

    def _set_param(
        self,
        layer_id,
        param_id,
        param_type,
        param,
    ):
        if isinstance(param, numpy.ndarray):
            param_temp = _Tensor.Ref('/tmp/rnn_param')
            param_temp.set_value(param)
            param = param_temp
        else: raise ValueError('Excepted a numpy array.')
        self.weights.expressions = dict() # Clear cached expressions
        outputs = RNNParamSet(
            inputs=[self.weights, param],
            layer_id=layer_id,
            param_id=param_id,
            param_type=param_type,
            rnn_mode=self.mode,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_directions=self.num_directions,
        )
        for k, v in outputs.expressions.items():
            _workspace.RunOperator(v)

    def _reset_params(self):
        numpy.random.seed(_cfg.GetRandomSeed())
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
                    matrix_shape = self._matrix_shape[packed_id][:]
                    bias_shape = self._bias_shape[packed_id][:]
                    matrix_shape[0] = bias_shape[0] = int(matrix_shape[0] / num_gates)
                    self._set_param(
                        layer_id=pseudo_layer_id,
                        param_id=param_id,
                        param_type='matrix',
                        param=matrix_init(matrix_shape),
                    )
                    self._set_param(
                        layer_id=pseudo_layer_id,
                        param_id=param_id,
                        param_type='bias',
                        param=bias_init(bias_shape),
                    )
        self.weights.expressions = weights_states
        self._init_params = True

    def create(
        self,
        x,
        hx=None,
        cx=None,
        required_hidden=True,
        required_cell=False,
    ):
        """Return outputs of this rnn.

        Parameters
        ----------
        x : Tensor
            The input tensor.
        hx : Tensor, optional
            The h(0) state.
        cx : Tensor, optional
            The c(0) state.
        required_hidden : bool, optional
            Return ``y`` and ``hidden`` if ``True``.
        required_hidden : bool, optional
            Return ``y``, ``hidden``, ``cell`` if ``True``.

        """
        if hx and not isinstance(hx, _Tensor):
            raise TypeError('Excepted hx as a Tensor, got {}.'.format(type(hx)))
        if cx and not isinstance(cx, _Tensor):
            raise TypeError('Excepted cx as a Tensor, got {}.'.format(type(cx)))

        if not self._init_params: self._reset_params()

        arguments = {
            'op_type': 'Recurrent',
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

        if required_cell: num_outputs = 3
        elif required_hidden: num_outputs = 2
        else: num_outputs = 1

        return _Tensor.CreateOperator(
            num_outputs=num_outputs, **arguments)

    def __call__(self, *args, **kwargs):
        return self.create(*args, **kwargs)