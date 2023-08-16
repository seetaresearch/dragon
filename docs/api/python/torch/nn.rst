vm.torch.nn
===========

.. only:: html

  Classes
  -------

  `class AdaptiveAvgPool1d <nn/AdaptiveAvgPool1d.html>`_
  : Apply 1d adaptive average pooling.

  `class AdaptiveAvgPool2d <nn/AdaptiveAvgPool2d.html>`_
  : Apply 2d adaptive average pooling.

  `class AdaptiveAvgPool3d <nn/AdaptiveAvgPool3d.html>`_
  : Apply 3d adaptive average pooling.

  `class AdaptiveMaxPool1d <nn/AdaptiveMaxPool1d.html>`_
  : Apply 1d adaptive max pooling.

  `class AdaptiveMaxPool2d <nn/AdaptiveMaxPool2d.html>`_
  : Apply 2d adaptive max pooling.

  `class AdaptiveMaxPool3d <nn/AdaptiveMaxPool3d.html>`_
  : Apply 3d adaptive max pooling.

  `class Affine <nn/Affine.html>`_
  : Apply affine transformation.

  `class AvgPool1d <nn/AvgPool1d.html>`_
  : Apply 1d average pooling.

  `class AvgPool2d <nn/AvgPool2d.html>`_
  : Apply 2d average pooling.

  `class AvgPool3d <nn/AvgPool3d.html>`_
  : Apply 3d average pooling.

  `class BatchNorm1d <nn/BatchNorm1d.html>`_
  : Apply batch normalization over 2d input.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `class BatchNorm2d <nn/BatchNorm2d.html>`_
  : Apply batch normalization over 3d input.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `class BatchNorm3d <nn/BatchNorm3d.html>`_
  : Apply batch normalization over 4d input.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `class BCEWithLogitsLoss <nn/BCEWithLogitsLoss.html>`_
  : Compute sigmoid cross entropy.

  `class ChannelShuffle <nn/ChannelShuffle.html>`_
  : Apply group shuffle to each channel.
  `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

  `class ConstantPad1d <nn/ConstantPad1d.html>`_
  : Pad input according to the last dimension with a constant.

  `class ConstantPad2d <nn/ConstantPad2d.html>`_
  : Pad input according to the last 2-dimensions with a constant.

  `class ConstantPad2d <nn/ConstantPad2d.html>`_
  : Pad input according to the last 3-dimensions with a constant.

  `class Conv1d <nn/Conv1d.html>`_
  : Apply 1d convolution.

  `class Conv2d <nn/Conv2d.html>`_
  : Apply 2d convolution.

  `class Conv3d <nn/Conv3d.html>`_
  : Apply 3d convolution.

  `class ConvTranspose1d <nn/ConvTranspose1d.html>`_
  : Apply 1d deconvolution.

  `class ConvTranspose2d <nn/ConvTranspose2d.html>`_
  : Apply 2d deconvolution.

  `class ConvTranspose3d <nn/ConvTranspose3d.html>`_
  : Apply 3d deconvolution.

  `class CosineSimilarity <nn/CosineSimilarity.html>`_
  : Compute softmax cross entropy.

  `class CrossEntropyLoss <nn/CrossEntropyLoss.html>`_
  : Compute softmax cross entropy.

  `class CTCLoss <nn/CTCLoss.html>`_
  : Compute ctc loss.
  `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

  `class DropBlock2d <nn/DropBlock2d.html>`_
  : Set blocks to zero randomly.
  `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

  `class Dropout <nn/Dropout.html>`_
  : Set elements to zero randomly.
  `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

  `class DropPath <nn/DropPath.html>`_
  : Set examples to zero randomly.
  `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

  `class ELU <nn/ELU.html>`_
  : Apply exponential linear unit.
  `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

  `class Flatten <nn/Flatten.html>`_
  : Flatten input dimensions.

  `class GELU <nn/GELU.html>`_
  : Apply gaussian error linear unit.
  `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

  `class GroupNorm <nn/GroupNorm.html>`_
  : Apply group normalization.
  `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

  `class GRU <nn/RNN.html>`_
  : Apply a multi-layer gated recurrent unit (GRU) RNN.
  `[Cho et.al, 2014] <https://arxiv.org/abs/1406.1078>`_.

  `class GumbelSoftmax <nn/GumbelSoftmax.html>`_
  : Apply gumbel softmax with a temperature.
  `[Jang et.al, 2016] <https://arxiv.org/abs/1611.01144>`_.

  `class Hardsigmoid <nn/Hardsigmoid.html>`_
  : Apply hard sigmoid function.

  `class Hardswish <nn/Hardswish.html>`_
  : Apply hard swish function.
  `[Howard et.al, 2019] <https://arxiv.org/abs/1905.02244>`_.

  `class Identity <nn/Identity.html>`_
  : Apply identity transformation.

  `class KLDivLoss <nn/KLDivLoss.html>`_
  : Compute Kullback-Leibler divergence.

  `class L1Loss <nn/L1Loss.html>`_
  : Compute element-wise absolute value difference.

  `class LeakyReLU <nn/LeakyReLU.html>`_
  : Apply leaky rectified linear unit.

  `class Linear <nn/Linear.html>`_
  : Apply linear transformation.

  `class LayerNorm <nn/LayerNorm.html>`_
  : Apply layer normalization.
  `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_

  `class LocalResponseNorm <nn/LocalResponseNorm.html>`_
  : Apply local response normalization.
  `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

  `class LogSoftmax <nn/LogSoftmax.html>`_
  : Apply logarithm softmax function.

  `class LSTM <nn/LSTM.html>`_
  : Apply a multi-layer long short-term memory (LSTM) RNN.
  `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

  `class LSTMCell <nn/LSTMCell.html>`_
  : Apply a long short-term memory (LSTM) cell.
  `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

  `class MaxPool1d <nn/MaxPool1d.html>`_
  : Apply 1d max pooling.

  `class MaxPool2d <nn/MaxPool2d.html>`_
  : Apply 2d max pooling.

  `class MaxPool3d <nn/MaxPool3d.html>`_
  : Apply 3d max pooling.

  `class Module <nn/Module.html>`_
  : The base class of modules.

  `class ModuleList <nn/ModuleList.html>`_
  : The list module container.

  `class MSELoss <nn/MSELoss.html>`_
  : Compute element-wise squared error.

  `class MultiheadAttention <nn/MultiheadAttention.html>`_
  : Apply multihead attention.
  `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

  `class NLLLoss <nn/NLLLoss.html>`_
  : Compute negative likelihood loss.

  `class Parameter <nn/Parameter.html>`_
  : A wrapped tensor considered to be a module parameter.

  `class PixelShuffle <nn/PixelShuffle.html>`_
  : Rearrange depth elements into pixels.

  `class PixelUnshuffle <nn/PixelUnshuffle.html>`_
  : Rearrange pixels into depth elements.

  `class PReLU <nn/PReLU.html>`_
  : Apply parametric rectified linear unit.
  `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

  `class ReflectionPad1d <nn/ReflectionPad1d.html>`_
  : Pad input according to the last dimension by reflecting boundary.

  `class ReflectionPad2d <nn/ReflectionPad2d.html>`_
  : Pad input according to the last 2-dimensions by reflecting boundary.

  `class ReflectionPad3d <nn/ReflectionPad3d.html>`_
  : Pad input according to the last 3-dimensions by reflecting boundary.

  `class ReLU <nn/ReLU.html>`_
  : Apply rectified linear unit.
  `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

  `class ReLU6 <nn/ReLU.html>`_
  : Apply clipped-6 rectified linear unit.
  `[Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_.

  `class ReplicationPad1d <nn/ReplicationPad1d.html>`_
  : Pad input according to the last dimension by replicating boundary.

  `class ReplicationPad2d <nn/ReplicationPad2d.html>`_
  : Pad input according to the last 2-dimensions by replicating boundary.

  `class ReplicationPad3d <nn/ReplicationPad3d.html>`_
  : Pad input according to the last 3-dimensions by replicating boundary.

  `class RNN <nn/RNN.html>`_
  : Apply a multi-layer Elman RNN.
  `[Elman, 1990] <https://doi.org/10.1016>`_.

  `class SELU <nn/SELU.html>`_
  : Apply scaled exponential linear unit.
  `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

  `class Sequential <nn/Sequential.html>`_
  : The sequential module container.

  `class Sigmoid <nn/Sigmoid.html>`_
  : Apply sigmoid function.

  `class SigmoidFocalLoss <nn/SigmoidFocalLoss.html>`_
  : Compute sigmoid focal loss.
  `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`__.

  `class SiLU <nn/SiLU.html>`_
  : Apply sigmoid linear unit.
  `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

  `class SmoothL1Loss <nn/SmoothL1Loss.html>`_
  : Compute element-wise error transited from L1 and L2.
  `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

  `class Softmax <nn/Softmax.html>`_
  : Apply softmax function.

  `class Tanh <nn/Tanh.html>`_
  : Apply tanh function.

  `class TransformerDecoder <nn/TransformerDecoder.html>`_
  : Standard transformer decoder.
  `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

  `class TransformerDecoderLayer <nn/TransformerDecoderLayer.html>`_
  : Layer for a standard transformer decoder.
  `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

  `class TransformerEncoder <nn/TransformerEncoder.html>`_
  : Standard transformer encoder.
  `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

  `class TransformerEncoderLayer <nn/TransformerEncoderLayer.html>`_
  : Layer for a standard transformer encoder.
  `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

  `class SyncBatchNorm <nn/SyncBatchNorm.html>`_
  : Apply sync batch normalization over input.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `class Unfold <nn/Unfold.html>`_
  : Extract sliding blocks.

  `class Upsample <nn/Upsample.html>`_
  : Upsample input via interpolating neighborhoods.

  `class UpsamplingBilinear2d <nn/UpsamplingBilinear2d.html>`_
  : Upsample input via bilinear interpolating.

  `class UpsamplingNearest2d <nn/UpsamplingNearest2d.html>`_
  : Upsample input via nearest interpolating.

  `class ZeroPad2d <nn/ZeroPad2d.html>`_
  : Pad input according to the last 2-dimensions with zeros.

.. toctree::
  :hidden:

  nn/AdaptiveAvgPool1d
  nn/AdaptiveAvgPool2d
  nn/AdaptiveAvgPool3d
  nn/AdaptiveMaxPool1d
  nn/AdaptiveMaxPool2d
  nn/AdaptiveMaxPool3d
  nn/Affine
  nn/AvgPool1d
  nn/AvgPool2d
  nn/AvgPool3d
  nn/BatchNorm1d
  nn/BatchNorm2d
  nn/BatchNorm3d
  nn/BCEWithLogitsLoss
  nn/ChannelShuffle
  nn/ConstantPad1d
  nn/ConstantPad2d
  nn/ConstantPad3d
  nn/Conv1d
  nn/Conv2d
  nn/Conv3d
  nn/ConvTranspose1d
  nn/ConvTranspose2d
  nn/ConvTranspose3d
  nn/CosineSimilarity
  nn/CrossEntropyLoss
  nn/CTCLoss
  nn/DropBlock2d
  nn/Dropout
  nn/DropPath
  nn/ELU
  nn/Flatten
  nn/GELU
  nn/GroupNorm
  nn/GRU
  nn/GumbelSoftmax
  nn/Hardsigmoid
  nn/Hardswish
  nn/Identity
  nn/KLDivLoss
  nn/L1Loss
  nn/LayerNorm
  nn/LeakyReLU
  nn/Linear
  nn/LocalResponseNorm
  nn/LogSoftmax
  nn/LSTM
  nn/LSTMCell
  nn/MaxPool1d
  nn/MaxPool2d
  nn/MaxPool3d
  nn/Module
  nn/ModuleList
  nn/MSELoss
  nn/MultiheadAttention
  nn/NLLLoss
  nn/Parameter
  nn/PixelShuffle
  nn/PixelUnshuffle
  nn/PReLU
  nn/ReflectionPad1d
  nn/ReflectionPad2d
  nn/ReflectionPad3d
  nn/ReLU
  nn/ReLU6
  nn/ReplicationPad1d
  nn/ReplicationPad2d
  nn/ReplicationPad3d
  nn/RNN
  nn/SELU
  nn/Sequential
  nn/Sigmoid
  nn/SigmoidFocalLoss
  nn/SiLU
  nn/SmoothL1Loss
  nn/Softmax
  nn/Tanh
  nn/TransformerDecoder
  nn/TransformerDecoderLayer
  nn/TransformerEncoder
  nn/TransformerEncoderLayer
  nn/SyncBatchNorm
  nn/Unfold
  nn/Upsample
  nn/UpsamplingBilinear2d
  nn/UpsamplingNearest2d
  nn/ZeroPad2d

.. raw:: html

  <style>
    h1:before {
      content: "Module: dragon.";
      color: #103d3e;
    }
  </style>
