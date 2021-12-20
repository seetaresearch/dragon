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
"""Transformer modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.activation import MultiheadAttention
from dragon.vm.torch.core.nn.modules.container import ModuleList
from dragon.vm.torch.core.nn.modules.dropout import Dropout
from dragon.vm.torch.core.nn.modules.linear import Linear
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.modules.normalization import LayerNorm


class TransformerDecoder(Module):
    """Standard transformer decoder.
    `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

    Examples:

    ```python
    memory = torch.ones(4, 2, 8)
    tgt = torch.ones(5, 2, 8)
    decoder = torch.nn.TransformerDecoder(d_model=8, nhead=2, num_layers=1)
    out = decoder(tgt, memory)
    ```

    """

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        norm=None,
        norm_first=False,
    ):
        """Create a ``TransformerDecoder``.

        Parameters
        ----------
        d_model : int
            The dimension of features.
        nhead : int
            The number of parallel heads.
        num_layers : int
            The number of stack layers.
        dim_feedforward : int, optional, default=2048
            The dimension of feedforward network.
        dropout: float, optional, default=0.1
            The dropout ratio.
        activation : str, optional, default='relu'
             The activation function.
        norm : torch.nn.Module, optional
            The norm module.
        norm_first : bool, optional, default=False
            Apply layer form before attention and feedforward.

        """
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            ) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderLayer(Module):
    """Layer for a standard transformer decoder .
    `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

    Examples:

    ```python
    memory = torch.ones(4, 2, 8)
    tgt = torch.ones(5, 2, 8)
    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=8, nhead=2)
    out = decoder_layer(tgt, memory)
    ```

    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        norm_first=False,
    ):
        """Create a ``TransformerDecoderLayer``.

        Parameters
        ----------
        d_model : int
            The dimension of features.
        nhead : int
            The number of parallel heads.
        dim_feedforward : int, optional, default=2048
            The dimension of feedforward network.
        dropout: float, optional, default=0.1
            The dropout ratio.
        activation : str, optional, default='relu'
             The activation function.
        norm_first : bool, optional, default=False
            Apply layer form before attention and feedforward.

        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, inplace=True)
        self.dropout2 = Dropout(dropout, inplace=True)
        self.dropout3 = Dropout(dropout, inplace=True)
        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        if self.norm_first:
            tgt2 = self.norm1(tgt)
            tgt2 = self.self_attn(
                tgt2, tgt2, tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False)[0]
            tgt2 = self.dropout1(tgt2)
            tgt = tgt2.__iadd__(tgt)
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(
                tgt2, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False)[0]
            tgt2 = self.dropout2(tgt2)
            tgt = tgt2.__iadd__(tgt)
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt2 = self.dropout3(tgt2)
            tgt = tgt2.__iadd__(tgt)
            return tgt

        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False)[0]
        tgt2 = self.dropout1(tgt2)
        tgt2 += tgt
        tgt = self.norm1(tgt2)
        tgt2 = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False)[0]
        tgt2 = self.dropout2(tgt2)
        tgt2 += tgt
        tgt = self.norm2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt2 = self.dropout3(tgt2)
        tgt2 += tgt
        tgt = self.norm3(tgt2)
        return tgt


class TransformerEncoder(Module):
    """Standard transformer encoder.
    `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

    Examples:

    ```python
    src = torch.ones(4, 2, 8)
    encoder = torch.nn.TransformerEncoder(d_model=8, nhead=2, num_layers=1)
    out = encoder(src)
    ```

    """

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        norm=None,
        norm_first=False,
    ):
        """Create a ``TransformerEncoder``.

        Parameters
        ----------
        d_model : int
            The dimension of features.
        nhead : int
            The number of parallel heads.
        num_layers : int
            The number of stack layers.
        dim_feedforward : int, optional, default=2048
            The dimension of feedforward network.
        dropout: float, optional, default=0.1
            The dropout ratio.
        activation : str, optional, default='relu'
             The activation function.
        norm : torch.nn.Module, optional
            The norm module.
        norm_first : bool, optional, default=False
            Apply layer form before attention and feedforward.

        """
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            ) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(Module):
    """Layer for a standard transformer encoder .
    `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

    Examples:

    ```python
    src = torch.ones(4, 2, 8)
    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=8, nhead=2)
    out = encoder_layer(src)
    ```

    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        norm_first=False,
    ):
        """Create a ``TransformerEncoderLayer``.

        Parameters
        ----------
        d_model : int
            The dimension of features.
        nhead : int
            The number of parallel heads.
        dim_feedforward : int, optional, default=2048
            The dimension of feedforward network.
        dropout: float, optional, default=0.1
            The dropout ratio.
        activation : str, optional, default='relu'
             The activation function.
        norm_first : bool, optional, default=False
            Apply layer form before attention and feedforward.

        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, inplace=True)
        self.dropout2 = Dropout(dropout, inplace=True)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.norm_first:
            src2 = self.norm1(src)
            src2 = self.self_attn(
                src2, src2, src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False)[0]
            src2 = self.dropout1(src2)
            src = src2.__iadd__(src)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src2 = self.dropout2(src2)
            src = src2.__iadd__(src)
            return src

        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False)[0]
        src2 = self.dropout1(src2)
        src2 += src
        src = self.norm1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        src2 += src
        src = self.norm2(src2)
        return src


def _get_activation_fn(activation):
    """Return the activation function."""
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError('Unknown activation: {}'.format(activation))
