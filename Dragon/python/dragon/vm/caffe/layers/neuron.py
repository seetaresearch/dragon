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

"""The Implementation of the neuron layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import ops as _ops
from ..layer import Layer as _Layer


class ReLULayer(_Layer):
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
            self.arguments = {'slope': param.negative_slope}

    def LayerSetup(self, bottom):
        return _ops.Relu(bottom, **self.arguments)


class PReLULayer(_Layer):
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
        self.arguments= {
            'channel_shared': param.channel_shared,
            'data_format': 'NCHW',
        }
        # Trainable slope
        self.AddBlob(filler=self.GetFiller(param, 'filler'), value=0.25)

    def LayerSetup(self, bottom):
        inputs = [bottom] + [blob['data'] for blob in self._blobs]
        return _ops.PRelu(inputs, **self.arguments)


class ELULayer(_Layer):
    """The implementation of ``ELULayer``.

    Parameters
    ----------
    alpha : float
        The alpha. Refer `ELUParameter.alpha`_.

    """
    def __init__(self, LayerParameter):
        super(ELULayer, self).__init__(LayerParameter)
        self.arguments = {'alpha': float(LayerParameter.elu_param.alpha)}

    def LayerSetup(self, bottom):
        return _ops.Elu(bottom, **self.arguments)


class SELULayer(_Layer):
    """The implementation of ``SELULayer``."""

    def __init__(self, LayerParameter):
        super(SELULayer, self).__init__(LayerParameter)

    def LayerSetup(self, bottom):
        return _ops.SElu(bottom, **self.arguments)


class SigmoidLayer(_Layer):
    """The implementation of ``SigmoidLayer``."""

    def __init__(self, LayerParameter):
        super(SigmoidLayer, self).__init__(LayerParameter)

    def LayerSetup(self, bottom):
        return _ops.Sigmoid(bottom, **self.arguments)


class TanHLayer(_Layer):
    """The implementation of ``TanHLayer``."""

    def __init__(self, LayerParameter):
        super(TanHLayer, self).__init__(LayerParameter)

    def LayerSetup(self, bottom):
        return _ops.Tanh(bottom, **self.arguments)


class DropoutLayer(_Layer):
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
        self.arguments = {
            'prob': param.dropout_ratio,
            'scale': param.scale_train \
                if hasattr(param, 'scale_train') else True,
        }

    def LayerSetup(self, bottom):
        return _ops.Dropout(bottom, **self.arguments)


class PowerLayer(_Layer):
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
        self.arguments = {
            'power': param.power,
            'scale': param.scale,
            'shift': param.shift,
        }

    def LayerSetup(self, bottom):
        return _ops.Pow(bottom, **self.arguments)