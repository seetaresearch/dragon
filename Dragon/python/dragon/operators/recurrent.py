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

from . import *
from .rnn.rnn_wrapper import RNNBase


class RNN(RNNBase):
    """Multi-layer Elman-RNN with `TanH` or `ReLU` non-linearity. `[Elman, 1990] <https://doi.org/10.1016>`_.

    The data format of inputs should be ``[T, N, C]``.

    Examples
    --------
    >>> rnn = RNN(32, 64, num_layers=1, bidirectional=True)
    >>> x = Tensor('x').Variable()
    >>> outputs, hidden = rnn(x)

    """
    def __init__(self, input_size, hidden_size, nonlinearity='relu',
                 num_layers=1, bidirectional=False, dropout=0, name=None):
        """Construct a RNN instance.

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
        bidirectional : bool
            Whether to use bidirectional rnn.
        dropout : number
            The dropout ratio. ``0`` means ``Disabled``.
        name : str, optional
            The optional name for weights.

        Returns
        -------
        RNNBase
            The wrapper of general RNN.

        """
        mode = 'rnn_relu' if nonlinearity == 'relu' else 'rnn_tanh'
        super(RNN, self).__init__(mode, input_size, hidden_size,
            num_layers, bidirectional, dropout, name)


class LSTM(RNNBase):
    """Multi-layer Long Short-Term Memory(LSTM) RNN. `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    The data format of inputs should be ``[T, N, C]``.

    Examples
    --------
    >>> rnn = LSTM(32, 64, num_layers=2, bidirectional=True, dropout=0.5)
    >>> x = Tensor('x').Variable()
    >>> outputs, hidden = rnn(x)

    """
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bidirectional=False, dropout=0, name=None):
        """Construct a LSTM instance.

        Parameters
        ----------
        input_size : int
            The dimension of inputs.
        hidden_size : int
            The dimension of hidden/outputs.
        num_layers : int
            The number of recurrent layers.
        bidirectional : bool
            Whether to use bidirectional rnn.
        dropout : number
            The dropout ratio. ``0`` means ``Disabled``.
        name : str, optional
            The optional name for weights.

        Returns
        -------
        RNNBase
            The wrapper of general RNN.

        """
        super(LSTM, self).__init__('lstm', input_size, hidden_size,
            num_layers, bidirectional, dropout, name)


class GRU(RNNBase):
    """Multi-layer Gated Recurrent Unit (GRU) RNN. `[Cho et.al, 2014] <https://arxiv.org/abs/1406.1078>`_.

    The data format of inputs should be ``[T, N, C]``.

    Examples
    --------
    >>> rnn = GRU(32, 64, num_layers=2, bidirectional=False)
    >>> x = Tensor('x').Variable()
    >>> outputs, hidden = rnn(x)

    """
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bidirectional=False, dropout=0, name=None):
        """Construct a GRU instance.

        Parameters
        ----------
        input_size : int
            The dimension of inputs.
        hidden_size : int
            The dimension of hidden/outputs.
        num_layers : int
            The number of recurrent layers.
        bidirectional : bool
            Whether to use bidirectional rnn.
        dropout : number
            The dropout ratio. ``0`` means ``Disabled``.
        name : str, optional
            The optional name for weights.

        Returns
        -------
        RNNBase
            The wrapper of general RNN.

        """
        super(GRU, self).__init__('gru', input_size, hidden_size,
            num_layers, bidirectional, dropout, name)


@OpSchema.Inputs(2)
def LSTMCell(inputs, **kwargs):
    """Single-layer Long Short-Term Memory(LSTM) Cell. `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    The data format of inputs should be ``[N, C]``.

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs, represent x(4-concatenated) and cx respectively.

    Returns
    -------
    sequence of Tensor
        The outputs, ``h`` and ``c`` respectively.

    """
    return Tensor.CreateOperator('LSTMCell', num_outputs=2, **ParseArgs(locals()))