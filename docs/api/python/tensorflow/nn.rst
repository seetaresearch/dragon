vm.tensorflow.nn
================

.. only:: html

  Functions
  ---------

  `avg_pool(...) <nn/avg_pool.html>`_
  : Apply the n-dimension average pooling.

  `avg_pool1d(...) <nn/avg_pool1d.html>`_
  : Apply the 1d average pooling.

  `avg_pool2d(...) <nn/avg_pool2d.html>`_
  : Apply the 2d average pooling.

  `avg_pool3d(...) <nn/avg_pool3d.html>`_
  : Apply the 3d average pooling.

  `batch_normalization(...) <nn/batch_normalization.html>`_
  : Apply the batch normalization.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `bias_add(...) <nn/bias_add.html>`_
  : Add the bias across channels to input.

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

  `convolution(...) <nn/convolution.html>`_
  : Apply the n-dimension convolution.

  `conv_transpose(...) <nn/conv_transpose.html>`_
  : Apply the n-dimension deconvolution.

  `depthwise_conv2d(...) <nn/depthwise_conv2d.html>`_
  : Apply the 2d depthwise convolution.
  `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.

  `depth_to_space(...) <nn/depth_to_space.html>`_
  : Rearrange depth data into spatial blocks.

  `dropout(...) <nn/dropout.html>`_
  : Set the elements of input to zero randomly.
  `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

  `elu(...) <nn/elu.html>`_
  : Apply the exponential exponential linear unit to input.
  `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

  `leaky_relu(...) <nn/leaky_relu.html>`_
  : Apply the leaky rectified linear unit.

  `local_response_normalization(...) <nn/local_response_normalization.html>`_
  : Apply the local response normalization.
  `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

  `log_softmax(...) <nn/log_softmax.html>`_
  : Apply the composite of logarithm and softmax.

  `max_pool(...) <nn/max_pool.html>`_
  : Apply the n-dimension max pooling.

  `max_pool1d(...) <nn/max_pool1d.html>`_
  : Apply the 1d max pooling.

  `max_pool2d(...) <nn/max_pool2d.html>`_
  : Apply the 2d max pooling.

  `max_pool3d(...) <nn/max_pool3d.html>`_
  : Apply the 3d max pooling.

  `moments(...) <nn/moments.html>`_
  : Compute the mean and variance of input along the given axes.

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
  : Apply the softmax function.

  `softmax_cross_entropy_with_logits(...) <nn/softmax_cross_entropy_with_logits.html>`_
  : Compute the softmax cross entropy with contiguous labels.

  `space_to_depth(...) <nn/space_to_depth.html>`_
  : Rearrange blocks of spatial data into depth.

  `sparse_softmax_cross_entropy_with_logits(...) <nn/sparse_softmax_cross_entropy_with_logits.html>`_
  : Compute the softmax cross entropy with sparse labels.

  `swish(...) <nn/swish.html>`_
  : Apply the swish function.
  `[Ramachandran et.al, 2017] <https://arxiv.org/abs/1710.05941>`_.

.. toctree::
  :hidden:

  nn/avg_pool
  nn/avg_pool1d
  nn/avg_pool2d
  nn/avg_pool3d
  nn/batch_normalization
  nn/bias_add
  nn/conv1d
  nn/conv1d_transpose
  nn/conv2d
  nn/conv2d_transpose
  nn/conv3d
  nn/conv3d_transpose
  nn/convolution
  nn/conv_transpose
  nn/depthwise_conv2d
  nn/depth_to_space
  nn/dropout
  nn/elu
  nn/leaky_relu
  nn/local_response_normalization
  nn/log_softmax
  nn/max_pool
  nn/max_pool1d
  nn/max_pool2d
  nn/max_pool3d
  nn/moments
  nn/relu
  nn/relu6
  nn/selu
  nn/softmax
  nn/softmax_cross_entropy_with_logits
  nn/space_to_depth
  nn/sparse_softmax_cross_entropy_with_logits
  nn/swish

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
