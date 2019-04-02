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

"""The Implementation of the data layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import ops as _ops
from ..layer import Layer as _Layer


class SoftmaxWithLossLayer(_Layer):
    """The implementation of ``SoftmaxWithLossLayer``.

    Parameters
    ----------
    axis : int
        The axis of softmax. Refer `SoftmaxParameter.axis`_.
    ignore_label : int
        The label id to ignore. Refer `LossParameter.ignore_label`_.
    normalization : NormalizationMode
        The normalization. Refer `LossParameter.normalization`_.
    normalize : boolean
        Whether to normalize. Refer `LossParameter.normalize`_.

    """
    def __init__(self, LayerParameter):
        super(SoftmaxWithLossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        softmax_param = LayerParameter.softmax_param
        norm_mode = {0: 'FULL', 1: 'VALID', 2: 'BATCH_SIZE', 3: 'NONE', 4: 'UNIT'}
        normalization = 'VALID'
        if param.HasField('normalize'):
            if not param.normalize: normalization = 'BATCH_SIZE'
        else:
            normalization = norm_mode[param.normalization]
        self.arguments = {
            'axis': softmax_param.axis,
            'normalization': normalization,
            'ignore_labels': [param.ignore_label]
                if param.HasField('ignore_label') else [],
        }

    def LayerSetup(self, bottom):
        loss = _ops.SparseSoftmaxCrossEntropy(bottom, **self.arguments)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss


class SigmoidCrossEntropyLossLayer(_Layer):
    """The implementation of ``SigmoidCrossEntropyLossLayer``.

    Parameters
    ----------
    normalization : NormalizationMode
        The normalization. Refer `LossParameter.normalization`_.
    normalize : boolean
        Whether to normalize. Refer `LossParameter.normalize`_.

    """
    def __init__(self, LayerParameter):
        super(SigmoidCrossEntropyLossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        norm_mode = {0: 'FULL', 1: 'VALID', 2: 'BATCH_SIZE', 3: 'NONE', 4: 'UNIT'}
        normalization = 'VALID'
        if param.HasField('normalize'):
            if not param.normalize: normalization = 'BATCH_SIZE'
        else: normalization = norm_mode[param.normalization]
        self.arguments = {'normalization': normalization}

    def LayerSetup(self, bottom):
        loss = _ops.SigmoidCrossEntropy(bottom, **self.arguments)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss


class L2LossLayer(_Layer):
    """The implementation of ``L2LossLayer``.

    Parameters
    ----------
    normalization : NormalizationMode
        The normalization. Refer `LossParameter.normalization`_.
    normalize : boolean
        Whether to normalize. Refer `LossParameter.normalize`_.

    """
    def __init__(self, LayerParameter):
        super(L2LossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        norm_mode = {0: 'FULL', 1: 'BATCH_SIZE', 2: 'BATCH_SIZE', 3: 'NONE'}
        normalization = 'BATCH_SIZE'
        if param.HasField('normalize'):
            if param.normalize: normalization = 'FULL'
        else: normalization = norm_mode[param.normalization]
        self.arguments = {'normalization': normalization}

    def LayerSetup(self, bottom):
        loss = _ops.L2Loss(bottom, **self.arguments)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss


class SmoothL1LossLayer(_Layer):
    """The implementation of ``SmoothL1LossLayer``.

    Parameters
    ----------
    sigma : float
        The sigma. Refer `SmoothL1LossParameter.sigma`_.
    normalization : NormalizationMode
        The normalization. Refer `LossParameter.normalization`_.
    normalize : boolean
        Whether to normalize. Refer `LossParameter.normalize`_.

    """
    def __init__(self, LayerParameter):
        super(SmoothL1LossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        smooth_l1_param = LayerParameter.smooth_l1_loss_param
        norm_mode = {0: 'FULL', 1: 'BATCH_SIZE', 2: 'BATCH_SIZE', 3: 'NONE'}
        normalization = 'BATCH_SIZE'
        if param.HasField('normalize'):
            if param.normalize: normalization = 'FULL'
        else: normalization = norm_mode[param.normalization]
        sigma2 = smooth_l1_param.sigma * smooth_l1_param.sigma
        self.arguments = {
            'beta': float(1. / sigma2),
            'normalization': normalization,
        }

    def LayerSetup(self, bottom):
        loss = _ops.SmoothL1Loss(bottom, **self.arguments)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss


class SigmoidWithFocalLossLayer(_Layer):
    """The implementation of ``SigmoidWithFocalLossLayer``.

    Parameters
    ----------
    axis : int
        The axis of softmax. Refer `SoftmaxParameter.axis`_.
    alpha : float
        The scale on the rare class. Refer `FocalLossParameter.alpha`_.
    gamma : float
        The exponential decay. Refer `FocalLossParameter.gamma`_.
    neg_id : int
        The negative id. Refer `FocalLossParameter.neg_id`_.
    normalization : NormalizationMode
        The normalization. Refer `LossParameter.normalization`_.
    normalize : boolean
        Whether to normalize. Refer `LossParameter.normalize`_.

    """
    def __init__(self, LayerParameter):
        super(SigmoidWithFocalLossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        softmax_param = LayerParameter.softmax_param
        focal_loss_param = LayerParameter.focal_loss_param
        norm_mode = {0: 'FULL', 1: 'VALID', 2: 'BATCH_SIZE', 3: 'NONE', 4: 'UNIT'}
        normalization = 'VALID'
        if param.HasField('normalize'):
            if not param.normalize: normalization = 'BATCH_SIZE'
        else: normalization = norm_mode[param.normalization]
        self.arguments = {
            'axis': softmax_param.axis,
            'normalization': normalization,
            'alpha': float(focal_loss_param.alpha),
            'gamma': float(focal_loss_param.gamma),
            'neg_id': focal_loss_param.neg_id,
        }

    def LayerSetup(self, bottom):
        loss = _ops.SigmoidFocalLoss(bottom, **self.arguments)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss


class SoftmaxWithFocalLossLayer(_Layer):
    """The implementation of ``SoftmaxWithFocalLossLayer``.

    Parameters
    ----------
    axis : int
        The axis of softmax. Refer `SoftmaxParameter.axis`_.
    alpha : float
        The scale on the rare class. Refer `FocalLossParameter.alpha`_.
    gamma : float
        The exponential decay. Refer `FocalLossParameter.gamma`_.
    neg_id : int
        The negative id. Refer `FocalLossParameter.neg_id`_.
    normalization : NormalizationMode
        The normalization. Refer `LossParameter.normalization`_.
    normalize : boolean
        Whether to normalize. Refer `LossParameter.normalize`_.

    """
    def __init__(self, LayerParameter):
        super(SoftmaxWithFocalLossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        softmax_param = LayerParameter.softmax_param
        focal_loss_param = LayerParameter.focal_loss_param
        norm_mode = {0: 'FULL', 1: 'VALID', 2: 'BATCH_SIZE', 3: 'NONE', 4: 'UNIT'}
        normalization = 'VALID'
        if param.HasField('normalize'):
            if not param.normalize: normalization = 'BATCH_SIZE'
        else: normalization = norm_mode[param.normalization]
        self.arguments = {
            'axis': softmax_param.axis,
            'normalization': normalization,
            'ignore_labels': [param.ignore_label] if param.HasField('ignore_label') else [],
            'alpha': float(focal_loss_param.alpha),
            'gamma': float(focal_loss_param.gamma),
            'neg_id': focal_loss_param.neg_id,
        }

    def LayerSetup(self, bottom):
        loss = _ops.SoftmaxFocalLoss(bottom, **self.arguments)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss