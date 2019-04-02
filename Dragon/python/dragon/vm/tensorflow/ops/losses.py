# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/losses/losses_impl.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import ops as _ops
from dragon.vm.tensorflow.framework import ops


class Reduction(object):
    """Types of loss reduction.

    Contains the following values:

    * `NONE`: Un-reduced weighted losses with the same shape as input.
    * `SUM`: Scalar sum of weighted losses.
    * `MEAN`: Scalar `SUM` divided by sum of weights. DEPRECATED.
    * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
    * `SUM_OVER_NONZERO_WEIGHTS`: Scalar `SUM` divided by number of non-zero
       weights. DEPRECATED.
    * `SUM_BY_NONZERO_WEIGHTS`: Same as `SUM_OVER_NONZERO_WEIGHTS`.

    """
    NONE = "none"
    SUM = "weighted_sum"
    SUM_OVER_BATCH_SIZE = "weighted_sum_over_batch_size"
    MEAN = "weighted_mean"
    SUM_BY_NONZERO_WEIGHTS = "weighted_sum_by_nonzero_weights"
    SUM_OVER_NONZERO_WEIGHTS = SUM_BY_NONZERO_WEIGHTS

    @classmethod
    def all(cls):
        return (
            cls.NONE,
            cls.SUM,
            cls.MEAN,
            cls.SUM_OVER_BATCH_SIZE,
            cls.SUM_OVER_NONZERO_WEIGHTS,
            cls.SUM_BY_NONZERO_WEIGHTS,
        )

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError("Invalid Reduction Key %s." % key)


def softmax_cross_entropy(
    onehot_labels,
    logits,
    weights=1.,
    label_smoothing=0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS,
):
    if onehot_labels is None: raise ValueError("onehot_labels must not be None.")
    if logits is None: raise ValueError("logits must not be None.")
    normalization = None
    if reduction == Reduction.NONE: normalization = 'UNIT'
    elif reduction == Reduction.MEAN: normalization = 'FULL'
    elif reduction == Reduction.SUM_BY_NONZERO_WEIGHTS or \
            reduction == Reduction.SUM_OVER_NONZERO_WEIGHTS:
        normalization = 'NONE'
    elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        normalization = 'BATCH_SIZE'
    loss = _ops.SoftmaxCrossEntropy(
        [logits, onehot_labels],
        normalization=normalization,
        name=scope,
    )
    if weights != 1.0: loss = weights * loss
    ops.add_to_collection(loss_collection, loss)
    return loss


def sparse_softmax_cross_entropy(
    labels,
    logits,
    weights=1.,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS,
):
    if labels is None: raise ValueError("labels must not be None.")
    if logits is None: raise ValueError("logits must not be None.")
    normalization = None
    if reduction == Reduction.NONE: normalization = 'UNIT'
    elif reduction == Reduction.MEAN: normalization = 'FULL'
    elif reduction == Reduction.SUM_BY_NONZERO_WEIGHTS or \
            reduction == Reduction.SUM_OVER_NONZERO_WEIGHTS:
        normalization = 'NONE'
    elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        normalization = 'BATCH_SIZE'
    loss = _ops.SparseSoftmaxCrossEntropy(
        [logits, labels],
        normalization=normalization,
        name=scope,
    )
    if weights != 1.0: loss = weights * loss
    ops.add_to_collection(loss_collection, loss)
    return loss