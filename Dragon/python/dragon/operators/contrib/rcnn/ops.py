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

from dragon.operators import *


@OpSchema.Inputs(3, INT_MAX)
def Proposal(inputs, strides, ratios, scales,
             pre_nms_top_n=6000, post_nms_top_n=300,
             nms_thresh=0.7, min_size=16,
             min_level=2, max_level=5,
             canonical_scale=224, canonical_level=4, **kwargs):
    """Generate Regional Proposals, introduced by `[Ren et.al, 2015] <https://arxiv.org/abs/1506.01497>`_.

    Multi-Level proposals was introduced by `[Lin et.al, 2017] <https://arxiv.org/abs/1612.03144>`_.

    For single level proposals(e.g. C4), the inputs should be: [cls_probs, bbox_deltas, im_info].

    For multiple level proposals(e.g. FPN), the inputs should be: [cls_score/Px, ...] + [cls_probs, bbox_deltas, im_info].

    Parameters
    ----------
    inputs : sequence of Tensor
        The inputs.
    strides : sequence of int
        The strides of anchors.
    ratios : sequence of float
        The ratios of anchors.
    scales : sequence of float
        The scales of anchors.
    pre_nms_top_n : int, optional, default=6000
        The number of anchors before nms.
    post_nms_top_n : int, optional, default=300
        The number of anchors after nms.
    nms_thresh : float, optional, default=0.7
        The threshold of nms.
    min_size : int, optional, default=16
        The min size of anchors.
    min_level : int, optional, default=2
        Finest level of the FPN pyramid.
    max_level : int, optional, default=5
        Coarsest level of the FPN pyramid.
    canonical_scale : int, optional, default=224
        The baseline scale of mapping policy.
    canonical_level : int, optional, default=4
        Heuristic level of the canonical scale.

    Returns
    -------
    Tensor
        The proposals.

    """
    arguments = ParseArgs(locals())
    num_levels = (max_level - min_level) + 1
    num_levels = 1 if len(inputs) == 3 else num_levels
    return Tensor.CreateOperator('Proposal', num_outputs=num_levels, **arguments)