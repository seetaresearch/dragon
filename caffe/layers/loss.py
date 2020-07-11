# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

"""The implementation of loss layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.ops import loss_ops
from dragon.vm.caffe.layer import Layer


class EuclideanLoss(Layer):
    r"""Compute the element-wise squared error.

    The ``EuclideanLoss`` function is defined as:

    .. math:: \text{L2Loss}(x, y) = 0.5(x - y)^{2}

    Examples:

    ```python
    layer {
      type: "EuclideanLoss"
      bottom: "bbox_pred"
      bottom: "bbox_target"
      top: "bbox_loss"
      loss_param {
        normalization: BATCH_SIZE
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(EuclideanLoss, self).__init__(layer_param)
        param = layer_param.loss_param
        norm_dict = {0: 'mean', 1: 'mean', 2: 'batch_size', 3: 'sum'}
        reduction = 'batch_size'
        if param.HasField('normalize'):
            if param.normalize:
                reduction = 'mean'
        else:
            reduction = norm_dict[param.normalization]
        self.arguments = {'reduction': reduction}

    def __call__(self, bottom):
        loss = loss_ops.l2_loss(bottom, **self.arguments)
        loss_weight = 1. if self.loss_weight is None else self.loss_weight
        return loss * (loss_weight * 0.5)


class SigmoidCrossEntropyLoss(Layer):
    r"""Compute the sigmoid cross entropy with contiguous targets.

    Examples:

    ```python
    layer {
      type: "SigmoidCrossEntropyLoss"
      bottom: "rpn_cls_score"
      bottom: "rpn_labels"
      top: "rpn_loss"
      loss_param {
        normalization: VALID
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(SigmoidCrossEntropyLoss, self).__init__(layer_param)
        param = layer_param.loss_param
        norm_dict = {0: 'mean', 1: 'valid', 2: 'batch_size', 3: 'sum'}
        reduction = 'valid'
        if param.HasField('normalize'):
            if not param.normalize:
                reduction = 'batch_size'
        else:
            reduction = norm_dict[param.normalization]
        self.arguments = {'reduction': reduction}

    def __call__(self, bottom):
        loss = loss_ops.sigmoid_cross_entropy(bottom, **self.arguments)
        if self.loss_weight is not None:
            loss *= self.loss_weight
        return loss


class SmoothL1Loss(Layer):
    r"""Compute the element-wise error transited from L1 and L2.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    Examples:

    ```python
    layer {
      type: "SmoothL1Loss"
      bottom: "bbox_pred"
      bottom: "bbox_targets"
      bottom: "bbox_inside_weights"
      bottom: "bbox_outside_weights"
      top: "bbox_loss"
      loss_param {
        normalization: BATCH_SIZE
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(SmoothL1Loss, self).__init__(layer_param)
        param = layer_param.loss_param
        smooth_l1_param = layer_param.smooth_l1_loss_param
        norm_dict = {0: 'mean', 1: 'mean', 2: 'batch_size', 3: 'sum'}
        reduction = 'batch_size'
        if param.HasField('normalize'):
            if param.normalize:
                reduction = 'mean'
        else:
            reduction = norm_dict[param.normalization]
        sigma2 = smooth_l1_param.sigma * smooth_l1_param.sigma
        self.arguments = {
            'beta': float(1. / sigma2),
            'reduction': reduction,
        }

    def __call__(self, bottom):
        loss = loss_ops.smooth_l1_loss(bottom, **self.arguments)
        if self.loss_weight is not None:
            loss *= self.loss_weight
        return loss


class SoftmaxWithLoss(Layer):
    r"""Compute the softmax cross entropy with sparse labels.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    layer {
      type: "SoftmaxWithLoss"
      bottom: "cls_score"
      bottom: "labels"
      top: "cls_loss"
      softmax_param {
        axis: 1
      }
      loss_param {
        ignore_label: -1
        normalization: VALID
      }
    }
    ```

    """

    def __init__(self, layer_param):
        super(SoftmaxWithLoss, self).__init__(layer_param)
        param = layer_param.loss_param
        softmax_param = layer_param.softmax_param
        norm_dict = {0: 'mean', 1: 'valid', 2: 'batch_size', 3: 'sum'}
        reduction = 'valid'
        if param.HasField('normalize'):
            if not param.normalize:
                reduction = 'batch_size'
        else:
            reduction = norm_dict[param.normalization]
        self.arguments = {
            'axis': softmax_param.axis,
            'reduction': reduction,
            'ignore_index': param.ignore_label
            if param.HasField('ignore_label') else None,
        }

    def __call__(self, bottom):
        loss = loss_ops.sparse_softmax_cross_entropy(bottom, **self.arguments)
        if self.loss_weight is not None:
            loss *= self.loss_weight
        return loss
