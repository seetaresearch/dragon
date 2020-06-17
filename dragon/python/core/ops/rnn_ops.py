# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
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

import itertools
import numpy
import warnings

from dragon.core.autograph.tensor import Tensor
from dragon.core.eager import context
from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import config
from dragon.core.ops import rnn_ops_lib
from dragon.core.ops.utils import OpSchema
from dragon.core.ops.utils import parse_args
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
        name=None,
    ):
        if dropout > 0 and num_layers == 1:
            warnings.warn("Add dropout to single-layer RNN is meaningless.")
        self.mode = mode.lower()
        self.num_gates = {'lstm': 4, 'gru': 3}.get(self.mode, 1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = float(dropout) if dropout > 0 else None
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.name = name
        self._register_parameters()
        self.reset_parameters()

    def forward(self, inputs, **kwargs):
        """Compute the output of RNN."""
        inputs = nest.flatten(inputs)
        op_lib = rnn_ops_lib.Recurrent
        if context.executing_eagerly():
            inputs.insert(1, self._weights)
            return op_lib \
                .instantiate(
                    mode=self.mode,
                    num_layers=self.num_layers,
                    hidden_size=self.hidden_size,
                    bidirectional=self.bidirectional,
                    dropout_ratio=self.dropout,
                    is_training=kwargs.get('is_training', False),
                ).apply(inputs)
        else:
            inputs.insert(1, self._weights_ref)
            return op_lib.blend(
                inputs=inputs,
                rnn_mode=self.mode,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                bidirectional=self.bidirectional,
                dropout_ratio=self.dropout,
            )

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
        self._param_set(li, pi, type, init_fn(param_shape))

    @property
    def weights(self):
        """Return the flatten weights of RNN module."""
        if context.executing_eagerly():
            return self._weights
        return self._weights_ref

    @classmethod
    def _orthogonal_init(cls, shape, gain=1, dtype='float32'):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_cols, num_rows) if num_rows < num_cols \
            else (num_rows, num_cols)
        W = numpy.random.randn(*flat_shape)
        q, r = numpy.linalg.qr(W)
        # Make Q uniform.
        d = numpy.diag(r)
        q *= numpy.sign(d)
        if num_rows < num_cols:
            q = q.T
        return gain * q.reshape(shape).astype(dtype)

    def _param_set(self, layer_id, param_id, param_type, param):
        if isinstance(param, numpy.ndarray):
            param = EagerTensor(param, copy=False)
        else:
            raise ValueError('Excepted a numpy array.')
        return rnn_ops_lib.RNNParamSet \
            .instantiate(
                layer_id=layer_id,
                param_id=param_id,
                param_type=param_type,
                mode=self.mode,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_directions=self.num_directions,
            ).apply([self._weights, param])

    def _register_parameters(self):
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
        self._weights = EagerTensor(shape=[self._weights_count], trainable=True)
        self._weights_ref = Tensor(self._weights.name)

    def _uniform_init(self, shape, dtype='float32'):
        stdv = 1. / numpy.sqrt(self.hidden_size)
        return numpy.random.uniform(-stdv, stdv, shape).astype(dtype)

    @classmethod
    def _zero_init(cls, shape, dtype='float32'):
        return numpy.zeros(shape, dtype=dtype)

    def __call__(self, *args, **kwargs):
        return self.forward(args, **kwargs)


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
def LSTMCell(inputs, **kwargs):
    r"""Apply a long short-term memory (LSTM) cell.
    `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    Parameters
    ----------
    inputs : Sequence[dragon.Tensor]
        The **x.4** and **cx**.

    Returns
    -------
    Sequence[dragon.Tensor]
        The **h** and **c**.

    """
    args = parse_args(locals())
    op_lib = rnn_ops_lib.LSTMCell
    if context.executing_eagerly():
        return op_lib.instantiate().apply(inputs)
    else:
        return op_lib.blend(num_outputs=2, **args)
