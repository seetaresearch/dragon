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
"""Sparse modules."""

from dragon.vm.torch.core.nn import functional
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
                    raise ValueError("<padding_idx> must be within <num_embeddings>.")
            elif padding_idx < 0:
                if padding_idx < -self.num_embeddings:
                    raise ValueError("<padding_idx> must be within <num_embeddings>.")
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.weight = Parameter(Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        if self.padding_idx is not None:
            self.weight[self.padding_idx] = 0

    def forward(self, input):
        return functional.embedding(input, self.weight, self.padding_idx)
