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
    inputs : list of Tensor
        The inputs.
    strides : list of int
        The strides of anchors.
    ratios : list of float
        The ratios of anchors.
    scales : list of float
        The scales of anchors.
    pre_nms_top_n : int
        The number of anchors before nms.
    post_nms_top_n : int
        The number of anchors after nms.
    nms_thresh : float
        The threshold of nms.
    min_size : int
        The min size of anchors.
    min_level : int
        Finest level of the FPN pyramid.
    max_level : int
        Coarsest level of the FPN pyramid.
    canonical_scale : int
        The baseline scale of mapping policy.
    canonical_level : int
        Heuristic level of the canonical scale.

    Returns
    -------
    Tensor
        The proposals.

    """
    CheckInputs(inputs, 3, INT_MAX)
    arguments = ParseArguments(locals())
    num_levels = (max_level - min_level) + 1
    num_levels = 1 if len(inputs) == 3 else num_levels
    outputs = Tensor.CreateOperator(nout=num_levels, op_type='Proposal', **arguments)
    return outputs