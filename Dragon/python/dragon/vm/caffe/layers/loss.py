# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.ops as ops

from .layer import Layer

class SoftmaxWithLossLayer(Layer):
    def __init__(self, LayerParameter):
        super(SoftmaxWithLossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        softmax_param = LayerParameter.softmax_param
        norm_mode = {0: 'FULL', 1: 'VALID', 2: 'BATCH_SIZE', 3: 'NONE'}
        normalization = 'VALID'
        if param.HasField('normalize'):
            if not param.normalize: normalization='BATCH_SIZE'
        else: normalization = norm_mode[param.normalization]
        self._param = {'axis': softmax_param.axis,
                       'normalization': normalization,
                       'ignore_labels': [param.ignore_label] if param.HasField('ignore_label') else [] }

    def Setup(self, bottom):
        super(SoftmaxWithLossLayer, self).Setup(bottom)
        return ops.SparseSoftmaxCrossEntropy(bottom, **self._param)


class SigmoidCrossEntropyLossLayer(Layer):
    def __init__(self, LayerParameter):
        super(SigmoidCrossEntropyLossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        norm_mode = {0: 'FULL', 1: 'FULL', 2: 'BATCH_SIZE', 3: 'NONE'}
        normalization = 'FULL'
        if param.HasField('normalize'):
            if not param.normalize: normalization = 'BATCH_SIZE'
        else: normalization = norm_mode[param.normalization]
        self._param = { 'normalization': normalization }

    def Setup(self, bottom):
        super(SigmoidCrossEntropyLossLayer, self).Setup(bottom)
        return ops.SigmoidCrossEntropy(bottom, **self._param)


class L2LossLayer(Layer):
    def __init__(self, LayerParameter):
        super(L2LossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        self._param = { 'normalize': param.normalize
            if param.HasField('normalize') else True }

    def Setup(self, bottom):
        super(L2LossLayer, self).Setup(bottom)
        return ops.L2Loss(bottom, **self._param)


class SmoothL1LossLayer(Layer):
    def __init__(self, LayerParameter):
        super(SmoothL1LossLayer, self).__init__(LayerParameter)
        param = LayerParameter.smooth_l1_loss_param
        self._param = {'sigma': float(param.sigma)}

    def Setup(self, bottom):
        super(SmoothL1LossLayer, self).Setup(bottom)
        return ops.SmoothL1Loss(bottom, **self._param)


class SoftmaxWithFocalLossLayer(Layer):
    def __init__(self, LayerParameter):
        super(SoftmaxWithFocalLossLayer, self).__init__(LayerParameter)
        param = LayerParameter.loss_param
        softmax_param = LayerParameter.softmax_param
        focal_loss_param = LayerParameter.focal_loss_param
        norm_mode = {0: 'FULL', 1: 'VALID', 2: 'BATCH_SIZE', 3: 'NONE'}
        normalization = 'VALID'
        if param.HasField('normalize'):
            if not param.normalize: normalization='BATCH_SIZE'
        else: normalization = norm_mode[param.normalization]
        self._param = {'axis': softmax_param.axis,
                       'normalization': normalization,
                       'ignore_labels': [param.ignore_label] if param.HasField('ignore_label') else [],
                       'alpha': float(focal_loss_param.alpha),
                       'gamma': float(focal_loss_param.gamma),
                       'eps': float(focal_loss_param.eps),
                       'use_pseudo_metric': focal_loss_param.use_pseudo_metric}

    def Setup(self, bottom):
        super(SoftmaxWithFocalLossLayer, self).Setup(bottom)
        return ops.SparseSoftmaxFocalLoss(bottom, **self._param)
