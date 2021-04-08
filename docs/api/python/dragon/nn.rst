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
  : Apply the batch normalization.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `bias_add(...) <nn/bias_add.html>`_
  : Add the bias across channels to input.

  `conv(...) <nn/conv.html>`_
  : Apply the n-dimension convolution.

  `conv_transpose(...) <nn/conv_transpose.html>`_
  : Apply the n-dimension deconvolution.

  `conv1d(...) <nn/conv1d.html>`_
  : Apply the 1d convolution.

  `conv1d_transpose(...) <nn/conv1d_transpose.html>`_
  : Apply the 1d deconvolution.

  `conv2d(...) <nn/conv2d.html>`_
  : Apply the 2d convolution.

  `conv2d_transpose(...) <nn/conv2d_transpose.html>`_
  : Apply the 2d deconvolution.

  `conv3d(...) <nn/conv3d.html>`_
  : Apply the 3d convolution.

  `conv3d_transpose(...) <nn/conv3d_transpose.html>`_
  : Apply the 3d deconvolution.

  `depthwise_conv2d(...) <nn/depthwise_conv2d.html>`_
  : Apply the 2d depthwise convolution.

  `depth_to_space(...) <nn/depth_to_space.html>`_
  : Rearrange depth data into spatial blocks.

  `dropout(...) <nn/dropout.html>`_
  : Set the elements of input to zero randomly.
  `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

  `drop_block(...) <nn/drop_block.html>`_
  : Set the blocks over input to zero randomly.
  `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

  `drop_path(...) <nn/drop_path.html>`_
  : Set the examples over input to zero randomly.
  `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

  `elu(...) <nn/elu.html>`_
  : Apply the exponential linear unit.
  `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

  `group_norm(...) <nn/group_norm.html>`_
  : Apply the group normalization.
  `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

  `hardsigmoid(...) <nn/hardsigmoid.html>`_
  : Apply the hard sigmoid function.

  `hardswish(...) <nn/hardswish.html>`_
  : Apply the hard swish function.
  `[Howard et.al, 2019] <https://arxiv.org/abs/1905.02244>`_.

  `instance_norm(...) <nn/instance_norm.html>`_
  : Apply the instance normalization.
  `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_

  `layer_norm(...) <nn/layer_norm.html>`_
  : Apply the layer normalization.
  `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_

  `leaky_relu(...) <nn/leaky_relu.html>`_
  : Apply the leaky rectified linear unit.

  `local_response_norm(...) <nn/local_response_norm.html>`_
  : Apply the local response normalization.
  `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

  `log_softmax(...) <nn/log_softmax.html>`_
  : Compute the composite of logarithm and softmax.

  `moments(...) <nn/moments.html>`_
  : Compute the mean and variance of input along the given axis.

  `prelu(...) <nn/prelu.html>`_
  : Apply the parametric rectified linear unit.
  `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

  `pool1d(...) <nn/pool1d.html>`_
  : Apply the 1d pooling.

  `pool2d(...) <nn/pool2d.html>`_
  : Apply the 2d pooling.

  `pool3d(...) <nn/pool3d.html>`_
  : Apply the 3d pooling.

  `relu(...) <nn/relu.html>`_
  : Apply the rectified linear unit.
  `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

  `relu6(...) <nn/relu6.html>`_
  : Apply the clipped-6 rectified linear unit.
  `[Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_.

  `selu(...) <nn/selu.html>`_
  : Apply the scaled exponential linear unit.
  `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

  `softmax(...) <nn/softmax.html>`_
  : Compute the softmax result.

  `space_to_depth(...) <nn/space_to_depth.html>`_
  : Rearrange blocks of spatial data into depth.

  `swish(...) <nn/swish.html>`_
  : Apply the swish function.
  `[Ramachandran et.al, 2017] <https://arxiv.org/abs/1710.05941>`_.
   
  `sync_batch_norm(...) <nn/sync_batch_norm.html>`_
  : Apply the batch normalization with synced statistics.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

.. toctree::
  :hidden:

  nn/GRU
  nn/LSTM
  nn/RNN
  nn/batch_norm
  nn/bias_add
  nn/conv
  nn/conv_transpose
  nn/conv1d
  nn/conv1d_transpose
  nn/conv2d
  nn/conv2d_transpose
  nn/conv3d
  nn/conv3d_transpose
  nn/depthwise_conv2d
  nn/depth_to_space
  nn/dropout
  nn/drop_block
  nn/drop_path
  nn/elu
  nn/group_norm
  nn/hardsigmoid
  nn/hardswish
  nn/instance_norm
  nn/layer_norm
  nn/leaky_relu
  nn/local_response_norm
  nn/log_softmax
  nn/moments
  nn/pool
  nn/pool1d
  nn/pool2d
  nn/pool3d
  nn/prelu
  nn/relu
  nn/relu6
  nn/selu
  nn/softmax
  nn/space_to_depth
  nn/swish
  nn/sync_batch_norm

.. raw:: html

  <style>
  h1:before {
    content: "Module: ";
    color: #103d3e;
  }
  </style>
