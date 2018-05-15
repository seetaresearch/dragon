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

import dragon.ops as ops

from ..layer import Layer


class SoftmaxWithLossLayer(Layer):
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
        else: normalization = norm_mode[param.normalization]
        self._param = {'axis': softmax_param.axis,
                       'normalization': normalization,
                       'ignore_labels': [param.ignore_label] if param.HasField('ignore_label') else [] }

    def Setup(self, bottom):
        super(SoftmaxWithLossLayer, self).Setup(bottom)
        loss = ops.SparseSoftmaxCrossEntropy(bottom, **self._param)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss


class SigmoidCrossEntropyLossLayer(Layer):
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
        self._param = {'normalization': normalization}

    def Setup(self, bottom):
        super(SigmoidCrossEntropyLossLayer, self).Setup(bottom)
        loss = ops.SigmoidCrossEntropy(bottom, **self._param)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss


class L2LossLayer(Layer):
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
        self._param = {'normalization': normalization}

    def Setup(self, bottom):
        super(L2LossLayer, self).Setup(bottom)
        loss = ops.L2Loss(bottom, **self._param)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss


class SmoothL1LossLayer(Layer):
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
        self._param = {'beta': float(1. / sigma2),
                       'normalization': normalization}

    def Setup(self, bottom):
        super(SmoothL1LossLayer, self).Setup(bottom)
        loss = ops.SmoothL1Loss(bottom, **self._param)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss


class SoftmaxWithFocalLossLayer(Layer):
    """The implementation of ``SoftmaxWithFocalLossLayer``.

    Parameters
    ----------
    axis : int
        The axis of softmax. Refer `SoftmaxParameter.axis`_.
    alpha : float
        The scale on the rare class. Refer `FocalLossParameter.alpha`_.
    gamma : float
        The exponential decay. Refer `FocalLossParameter.gamma`_.
    eps : float
        The eps. Refer `FocalLossParameter.eps`_.
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
        self._param = {'axis': softmax_param.axis,
                       'normalization': normalization,
                       'ignore_labels': [param.ignore_label] if param.HasField('ignore_label') else [],
                       'alpha': float(focal_loss_param.alpha),
                       'gamma': float(focal_loss_param.gamma),
                       'eps': float(focal_loss_param.eps),
                       'neg_id': focal_loss_param.neg_id}

    def Setup(self, bottom):
        super(SoftmaxWithFocalLossLayer, self).Setup(bottom)
        loss = ops.SparseSoftmaxFocalLoss(bottom, **self._param)
        if self._loss_weight is not None: loss *= self._loss_weight
        return loss