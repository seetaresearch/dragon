# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from layer import Layer
import dragon.ops as ops

class ReLULayer(Layer):
    def __init__(self, LayerParameter):
        super(ReLULayer, self).__init__(LayerParameter)
        param = LayerParameter.relu_param
        if param.HasField('negative_slope'):
            self._param = { 'slope': param.negative_slope }

    def Setup(self, bottom):
        super(ReLULayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Relu(input, **self._param)


class TanhLayer(Layer):
    def __init__(self, LayerParameter):
        super(TanhLayer, self).__init__(LayerParameter)

    def Setup(self, bottom):
        super(TanhLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Tanh(input, **self._param)


class DropoutLayer(Layer):
    def __init__(self, LayerParameter):
        super(DropoutLayer, self).__init__(LayerParameter)
        param = LayerParameter.dropout_param
        self._param = {'prob': param.dropout_ratio,
                       'scale': param.scale_train \
                            if hasattr(param, 'scale_train') else True}

    def Setup(self, bottom):
        super(DropoutLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Dropout(input, **self._param)


class PowerLayer(Layer):
    def __init__(self, LayerParameter):
        super(PowerLayer, self).__init__(LayerParameter)
        param = LayerParameter.power_param
        self._param = {'power': param.power,
                       'scale': param.scale,
                       'shift': param.shift}

    def Setup(self, bottom):
        super(PowerLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Pow(input, **self._param)