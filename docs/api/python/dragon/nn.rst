dragon.nn
=========

.. only:: html

  Classes
  -------

  `class GRU <nn/RNN.html>`_
  : Apply a multi-layer gated recurrent unit (GRU) RNN.
  `[Cho et.al, 2014] <https://arxiv.org/abs/1406.1078>`_.

  `class LSTM <nn/LSTM.html>`_
  : Apply a multi-layer long short-term memory (LSTM) RNN.
  `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.

  `class RNN <nn/RNN.html>`_
  : Apply a multi-layer Elman RNN.
  `[Elman, 1990] <https://doi.org/10.1016>`_.

  Functions
  ---------

  `batch_norm(...) <nn/batch_norm.html>`_
  : Apply batch normalization.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `bias_add(...) <nn/bias_add.html>`_
  : Add the bias across channels to input.

  `channel_norm(...) <nn/channel_norm.html>`_
  : Apply normalization to each channel of input.

  `channel_shuffle(...) <nn/channel_shuffle.html>`_
  : Apply group shuffle to each channel of input.
  `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

  `conv(...) <nn/conv.html>`_
  : Apply n-dimension convolution.

  `conv_transpose(...) <nn/conv_transpose.html>`_
  : Apply n-dimension deconvolution.

  `conv1d(...) <nn/conv1d.html>`_
  : Apply 1d convolution.

  `conv1d_transpose(...) <nn/conv1d_transpose.html>`_
  : Apply 1d deconvolution.

  `conv2d(...) <nn/conv2d.html>`_
  : Apply 2d convolution.

  `conv2d_transpose(...) <nn/conv2d_transpose.html>`_
  : Apply 2d deconvolution.

  `conv3d(...) <nn/conv3d.html>`_
  : Apply 3d convolution.

  `conv3d_transpose(...) <nn/conv3d_transpose.html>`_
  : Apply 3d deconvolution.

  `depth_to_space(...) <nn/depth_to_space.html>`_
  : Rearrange depth data into spatial blocks.

  `dropout(...) <nn/dropout.html>`_
  : Set elements of input to zero randomly.
  `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

  `drop_block(...) <nn/drop_block.html>`_
  : Set blocks of input to zero randomly.
  `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

  `drop_path(...) <nn/drop_path.html>`_
  : Set examples of input to zero randomly.
  `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

  `elu(...) <nn/elu.html>`_
  : Apply exponential linear unit.
  `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

  `gelu(...) <nn/gelu.html>`_
  : Apply gaussian error linear unit.
  `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

  `group_norm(...) <nn/group_norm.html>`_
  : Apply group normalization.
  `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

  `hardsigmoid(...) <nn/hardsigmoid.html>`_
  : Apply hard sigmoid function.

  `hardswish(...) <nn/hardswish.html>`_
  : Apply hard swish function.
  `[Howard et.al, 2019] <https://arxiv.org/abs/1905.02244>`_.

  `instance_norm(...) <nn/instance_norm.html>`_
  : Apply instance normalization.
  `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_

  `layer_norm(...) <nn/layer_norm.html>`_
  : Apply layer normalization.
  `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_

  `leaky_relu(...) <nn/leaky_relu.html>`_
  : Apply leaky rectified linear unit.

  `local_response_norm(...) <nn/local_response_norm.html>`_
  : Apply local response normalization.
  `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

  `log_softmax(...) <nn/log_softmax.html>`_
  : Apply logarithm softmax function.

  `lp_norm(...) <nn/lp_norm.html>`_
  : Apply lp normalization.

  `moments(...) <nn/moments.html>`_
  : Compute the mean and variance of input along the given axis.

  `prelu(...) <nn/prelu.html>`_
  : Apply parametric rectified linear unit.
  `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

  `pool1d(...) <nn/pool1d.html>`_
  : Apply 1d pooling.

  `pool2d(...) <nn/pool2d.html>`_
  : Apply 2d pooling.

  `pool3d(...) <nn/pool3d.html>`_
  : Apply 3d pooling.

  `relu(...) <nn/relu.html>`_
  : Apply rectified linear unit.
  `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

  `relu6(...) <nn/relu6.html>`_
  : Apply clipped-6 rectified linear unit.
  `[Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_.

  `selu(...) <nn/selu.html>`_
  : Apply scaled exponential linear unit.
  `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

  `silu(...) <nn/silu.html>`_
  : Apply sigmoid linear unit.
  `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

  `softmax(...) <nn/softmax.html>`_
  : Apply softmax function.

  `space_to_depth(...) <nn/space_to_depth.html>`_
  : Rearrange blocks of spatial data into depth.
   
  `sync_batch_norm(...) <nn/sync_batch_norm.html>`_
  : Apply batch normalization with synced statistics.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

.. toctree::
  :hidden:

  nn/GRU
  nn/LSTM
  nn/RNN
  nn/batch_norm
  nn/bias_add
  nn/channel_norm
  nn/channel_shuffle
  nn/conv
  nn/conv_transpose
  nn/conv1d
  nn/conv1d_transpose
  nn/conv2d
  nn/conv2d_transpose
  nn/conv3d
  nn/conv3d_transpose
  nn/depth_to_space
  nn/dropout
  nn/drop_block
  nn/drop_path
  nn/elu
  nn/gelu
  nn/group_norm
  nn/hardsigmoid
  nn/hardswish
  nn/instance_norm
  nn/layer_norm
  nn/leaky_relu
  nn/local_response_norm
  nn/log_softmax
  nn/lp_norm
  nn/moments
  nn/pool
  nn/pool1d
  nn/pool2d
  nn/pool3d
  nn/prelu
  nn/relu
  nn/relu6
  nn/selu
  nn/silu
  nn/softmax
  nn/space_to_depth
  nn/sync_batch_norm

.. raw:: html

  <style>
  h1:before {
    content: "Module: ";
    color: #103d3e;
  }
  </style>
