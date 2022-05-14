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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm import torch


def roi_align(
    input,
    boxes,
    output_size,
    spatial_scale=1.0,
    sampling_ratio=-1,
    aligned=False,
):
    r"""Apply the average roi align to input.
    `[He et.al, 2017] <https://arxiv.org/abs/1703.06870>`_.

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
    sampling_ratio : int, optional, default=-1
        The number of sampling grids for ``boxes``.
    aligned : bool, optional, default=False
        Whether to shift the input coordinates by ``-0.5``.

    Returns
    -------
    dragon.vm.torch.Tensor
        The output tensor.

    """
    return torch.autograd.Function.apply(
        'RoiAlign', input.device, [input, boxes],
        pooled_h=output_size[0], pooled_w=output_size[1],
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio, aligned=aligned)
