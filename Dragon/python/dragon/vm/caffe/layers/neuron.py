# --------------------------------------------------------
# Caffe @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.ops as ops
from dragon.core.tensor import Tensor

from ..layer import Layer

class ReLULayer(Layer):
    """The implementation of ``ReLULayer``.

    Parameters
    ----------
    negative_slope : float
        The slope of negative side. Refer `ReLUParameter.negative_slope`_.

    """
    def __init__(self, LayerParameter):
        super(ReLULayer, self).__init__(LayerParameter)
        param = LayerParameter.relu_param
        if param.HasField('negative_slope'):
            self._param = {'slope': param.negative_slope}

    def Setup(self, bottom):
        super(ReLULayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Relu(input, **self._param)


class PReLULayer(Layer):
    """The implementation of ``PReLULayer``.

    Parameters
    ----------
    filler : FillerParameter
        The filler of parameter(slope). Refer `PReLUParameter.filler`_.
    channel_shared : boolean
       Whether to share the parameter across channels. Refer `PReLUParameter.channel_shared`_.

    """
    def __init__(self, LayerParameter):
        super(PReLULayer, self).__init__(LayerParameter)
        param = LayerParameter.prelu_param
        self._param = {'channel_shared': param.channel_shared,
                       'data_format': 'NCHW'}
        slope = Tensor(LayerParameter.name + '@param0')
        slope_diff = Tensor(LayerParameter.name + '@param0_grad')
        if param.HasField('filler'):
            self.Fill(slope, param, 'filler')
        else:
            slope.Constant(value=0.25)
        self._blobs.append({'data': slope, 'diff': slope_diff})

    def Setup(self, bottom):
        super(PReLULayer, self).Setup(bottom)
        return ops.PRelu(bottom + [blob['data'] for blob in self._blobs], **self._param)


class ELULayer(Layer):
    """The implementation of ``ELULayer``.

    Parameters
    ----------
    alpha : float
        The alpha. Refer `ELUParameter.alpha`_.

    """
    def __init__(self, LayerParameter):
        super(ELULayer, self).__init__(LayerParameter)
        param = LayerParameter.elu_param
        self._param = {'alpha': float(param.alpha)}

    def Setup(self, bottom):
        super(ELULayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Elu(input, **self._param)


class SELULayer(Layer):
    """
    The implementation of ``SELULayer``.
    """
    def __init__(self, LayerParameter):
        super(SELULayer, self).__init__(LayerParameter)

    def Setup(self, bottom):
        super(SELULayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.SElu(input, **self._param)


class SigmoidLayer(Layer):
    """
    The implementation of ``SigmoidLayer``.
    """
    def __init__(self, LayerParameter):
        super(SigmoidLayer, self).__init__(LayerParameter)

    def Setup(self, bottom):
        super(SigmoidLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Sigmoid(input, **self._param)


class TanHLayer(Layer):
    """
    The implementation of ``TanHLayer``.
    """
    def __init__(self, LayerParameter):
        super(TanHLayer, self).__init__(LayerParameter)

    def Setup(self, bottom):
        super(TanHLayer, self).Setup(bottom)
        input = bottom[0] if isinstance(bottom, list) else bottom
        return ops.Tanh(input, **self._param)


class DropoutLayer(Layer):
    """The implementation of ``DropoutLayer``.

    Parameters
    ----------
    dropout_ratio : float
        The prob of dropping. Refer `DropoutParameter.dropout_ratio`_.
    scale_train : boolean
        Whether to scale the output. Refer `DropoutParameter.scale_train`_.

    """
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
    """The implementation of ``PowerLayer``.

    Parameters
    ----------
    power : float
         The power factor. Refer `PowerParameter.power`_.
    scale : float
         The scale factor. Refer `PowerParameter.scale`_.
    shift : float
         The shift magnitude. Refer `PowerParameter.shift`_.

    """
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