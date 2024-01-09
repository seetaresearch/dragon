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
"""Fold modules."""

from dragon.vm.torch.core.nn import functional
from dragon.vm.torch.core.nn.modules.module import Module


class Unfold(Module):
    r"""Extract sliding blocks.

    This module excepts the input size :math:`(N, C, D1, D2, ...)`,
    and output size is :math:`(N, C \times \prod(\text{kernel\_size}), L)`,
    where :math:`N` is the batch size, :math:`C` is the number of channels,
    :math:`L` is :math:`\prod(D_{\text{out}})`.

    Examples:

    ```python
    m = torch.nn.Unfold(3, padding=1)
    x = torch.ones(2, 2, 2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.unfold(...)`_

    """

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        """Create a ``Unfold`` module.

        Parameters
        ----------
        kernel_size : Union[int, Sequence[int]]
            The size of sliding window.
        dilation : Union[int, Sequence[int]], optional, default=1
            The dilated rate of sliding convolution.
        stride : Union[int, Sequence[int]], optional, default=1
            The stride of sliding window.
        padding : Union[int, Sequence[int]], optional, default=0
            The zero padding size.

        """
        super(Unfold, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def extra_repr(self):
        return (
            "kernel_size={kernel_size}, "
            "dilation={dilation}, "
            "padding={padding}, "
            "stride={stride}".format(**self.__dict__)
        )

    def forward(self, input):
        return functional.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
