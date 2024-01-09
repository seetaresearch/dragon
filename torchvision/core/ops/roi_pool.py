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
"""RoIPool functions."""

from dragon.vm import torch


def roi_pool(input, boxes, output_size, spatial_scale=1.0):
    r"""Apply the max roi pooling to input.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    The ``boxes`` should be packed with a shape like :math:`(N, 5)`,
    where :math:`N` is the number of boxes.

    Each box is a 5d sequence containing **(batch_index, x1, y1, x2, y2)**.

    Parameters
    ----------
    input : dragon.vm.torch.Tensor
        The input tensor.
    boxes : dragon.vm.torch.Tensor
        The box coordinates.
    output_size : Sequence[int]
        The output height and width.
    spatial_scale : float, optional, default=1.0
        The input scale to the size of ``boxes``.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return torch.autograd.Function.apply(
        "RoiPool",
        input.device,
        [input, boxes],
        pooled_h=output_size[0],
        pooled_w=output_size[1],
        spatial_scale=spatial_scale,
    )
