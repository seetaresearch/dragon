vm.torch.nn
===========

.. only:: html

    Classes
    -------

    `class Affine <nn/Affine.html>`_
    : Apply the affine transformation over input.

    `class AvgPool2d <nn/AvgPool2d.html>`_
    : Apply the 2d average pooling.

    `class BatchNorm1d <nn/BatchNorm1d.html>`_
    : Apply the batch normalization over 2d input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    `class BatchNorm2d <nn/BatchNorm2d.html>`_
    : Apply the batch normalization over 3d input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    `class BatchNorm3d <nn/BatchNorm3d.html>`_
    : Apply the batch normalization over 4d input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    `class BCEWithLogitsLoss <nn/BCEWithLogitsLoss.html>`_
    : Compute the sigmoid cross entropy with contiguous targets.

    `class ConstantPad1d <nn/ConstantPad1d.html>`_
    : Pad input according to the last dimension with a constant.

    `class ConstantPad2d <nn/ConstantPad2d.html>`_
    : Pad input according to the last 2-dimensions with a constant.

    `class ConstantPad2d <nn/ConstantPad2d.html>`_
    : Pad input according to the last 3-dimensions with a constant.

    `class Conv2d <nn/Conv2d.html>`_
    : Apply the 2d convolution.

    `class ConvTranspose2d <nn/ConvTranspose2d.html>`_
    : Apply the 2d deconvolution.

    `class CrossEntropyLoss <nn/CrossEntropyLoss.html>`_
    : Compute the softmax cross entropy with sparse labels.

    `class CTCLoss <nn/CTCLoss.html>`_
    : Compute the ctc loss with batched labels.
    `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

    `class DepthwiseConv2d <nn/DepthwiseConv2d.html>`_
    : Apply the 2d depthwise convolution.
    `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.

    `class DropBlock2d <nn/DropBlock2d.html>`_
    : Set the spatial blocks to zero randomly.
    `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

    `class Dropout <nn/Dropout.html>`_
    : Set the elements to zero randomly.
    `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

    `class DropPath <nn/DropPath.html>`_
    : Set the examples over input to zero randomly.
    `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

    `class ELU <nn/ELU.html>`_
    : Apply the exponential linear unit.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    `class GroupNorm <nn/GroupNorm.html>`_
    : Apply the group normalization.
    `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

    `class GRU <nn/RNN.html>`_
    : Apply a multi-layer gated recurrent unit (GRU) RNN.
    `[Cho et.al, 2014] <https://arxiv.org/abs/1406.1078>`_.

    `class GumbelSoftmax <nn/GumbelSoftmax.html>`_
    : Apply the gumbel softmax with a temperature.
    `[Jang et.al, 2016] <https://arxiv.org/abs/1611.01144>`_.

    `class L1Loss <nn/L1Loss.html>`_
    : Compute the element-wise absolute value difference.

    `class LeakyReLU <nn/LeakyReLU.html>`_
    : Apply the leaky rectified linear unit.

    `class Linear <nn/Linear.html>`_
    : Apply the linear transformation.

    `class LocalResponseNorm <nn/LocalResponseNorm.html>`_
    : Apply the local response normalization.
    `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

    `class LogSoftmax <nn/LogSoftmax.html>`_
    : Apply the composite of logarithm and softmax.

    `class LSTM <nn/LSTM.html>`_
    : Apply a multi-layer long short-term memory (LSTM) RNN.
    `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    `class LSTMCell <nn/LSTMCell.html>`_
    : Apply a long short-term memory (LSTM) cell.
    `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

    `class MaxPool2d <nn/MaxPool2d.html>`_
    : Apply the 2d MaxPool2d pooling.

    `class Module <nn/Module.html>`_
    : The base class of modules.

    `class MSELoss <nn/MSELoss.html>`_
    : Compute the element-wise squared error.

    `class NLLLoss <nn/NLLLoss.html>`_
    : Compute the negative likelihood loss with sparse labels.

    `class Parameter <nn/Parameter.html>`_
    : A wrapped tensor considered to be a module parameter.

    `class PReLU <nn/PReLU.html>`_
    : Apply the parametric rectified linear unit.
    `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

    `class ReflectionPad1d <nn/ReflectionPad1d.html>`_
    : Pad input according to the last dimension by reflecting boundary.

    `class ReflectionPad2d <nn/ReflectionPad2d.html>`_
    : Pad input according to the last 2-dimensions by reflecting boundary.

    `class ReflectionPad3d <nn/ReflectionPad3d.html>`_
    : Pad input according to the last 3-dimensions by reflecting boundary.

    `class ReLU <nn/ReLU.html>`_
    : Apply the rectified linear unit.
    `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    `class ReLU6 <nn/ReLU.html>`_
    : Apply the clipped-6 rectified linear unit.
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

    `class SCEWithLogitsLoss <nn/SCEWithLogitsLoss.html>`_
    : Compute the softmax cross entropy with contiguous targets.

    `class SELU <nn/SELU.html>`_
    : Apply the scaled exponential linear unit.
    `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    `class Sigmoid <nn/Sigmoid.html>`_
    : Apply the sigmoid function.

    `class SigmoidFocalLoss <nn/SigmoidFocalLoss.html>`_
    : Compute the sigmoid focal loss with sparse labels.
    `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`__.

    `class SmoothL1Loss <nn/SmoothL1Loss.html>`_
    : Compute the element-wise error transited from L1 and L2.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    `class Softmax <nn/Softmax.html>`_
    : Apply the softmax function.

    `class Tanh <nn/Tanh.html>`_
    : Apply the tanh function.

    `class SyncBatchNorm <nn/SyncBatchNorm.html>`_
    : Apply the sync batch normalization over input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

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

    nn/Affine
    nn/AvgPool2d
    nn/BatchNorm1d
    nn/BatchNorm2d
    nn/BatchNorm3d
    nn/BCEWithLogitsLoss
    nn/ConstantPad1d
    nn/ConstantPad2d
    nn/ConstantPad3d
    nn/Conv2d
    nn/ConvTranspose2d
    nn/CrossEntropyLoss
    nn/CTCLoss
    nn/DepthwiseConv2d
    nn/DropBlock2d
    nn/Dropout
    nn/DropPath
    nn/ELU
    nn/GroupNorm
    nn/GRU
    nn/GumbelSoftmax
    nn/L1Loss
    nn/LeakyReLU
    nn/Linear
    nn/LocalResponseNorm
    nn/LogSoftmax
    nn/LSTM
    nn/LSTMCell
    nn/MaxPool2d
    nn/Module
    nn/MSELoss
    nn/NLLLoss
    nn/Parameter
    nn/PReLU
    nn/ReflectionPad1d
    nn/ReflectionPad2d
    nn/ReflectionPad3d
    nn/ReLU
    nn/ReLU6
    nn/ReplicationPad1d
    nn/ReplicationPad2d
    nn/ReplicationPad3d
    nn/SCEWithLogitsLoss
    nn/RNN
    nn/SELU
    nn/Sigmoid
    nn/SigmoidFocalLoss
    nn/SmoothL1Loss
    nn/Softmax
    nn/Tanh
    nn/SyncBatchNorm
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
