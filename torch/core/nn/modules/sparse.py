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
"""Sparse modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn import init
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.tensor import Tensor


class Embedding(Module):
    """Lookup the embeddings of a fixed dictionary."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        """Create an ``Embedding`` module.

        Parameters
        ----------
        num_embeddings : int
            The dictionary size.
        embedding_dim : int
            The embedding dimension.
        padding_idx : int, optional
            The position where to return zeros.

        """
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                if padding_idx >= self.num_embeddings:
                    raise ValueError('<padding_idx> must be within <num_embeddings>.')
            elif padding_idx < 0:
                if padding_idx < -self.num_embeddings:
                    raise ValueError('<padding_idx> must be within <num_embeddings>.')
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.weight = Parameter(Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        if self.padding_idx is not None:
            self.weight[self.padding_idx] = 0

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx)
