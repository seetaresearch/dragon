vm.torch.nn.functional
======================

.. only:: html

  Functions
  ---------

  `avg_pool2d(...) <functional/avg_pool2d.html>`_
  : Apply the 2d average pooling to input.

  `batch_norm(...) <functional/batch_norm.html>`_
  : Apply the batch normalization to input.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `binary_cross_entropy_with_logits(...) <functional/binary_cross_entropy_with_logits.html>`_
  : Compute the sigmoid cross entropy with contiguous target.

  `conv2d(...) <functional/conv2d.html>`_
  : Apply 2d convolution to the input.

  `conv_transpose2d(...) <functional/conv_transpose2d.html>`_
  : Apply 2d deconvolution to the input.

  `cross_entropy(...) <functional/cross_entropy.html>`_
  : Compute the softmax cross entropy with sparse labels.

  `ctc_loss(...) <functional/ctc_loss.html>`_
  : Compute the ctc loss with batched labels.
  `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

  `depthwise_conv2d(...) <functional/depthwise_conv2d.html>`_
  : Apply 2d depthwise convolution to the input.
  `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.

  `drop_block2d(...) <functional/drop_block2d.html>`_
  : Set the spatial blocks over input to zero randomly.
  `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

  `drop_path(...) <functional/drop_path.html>`_
  : Set the examples over input to zero randomly.
  `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

  `dropout(...) <functional/dropout.html>`_
  : Set the elements of the input to zero randomly.
  `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

  `elu(...) <functional/elu.html>`_
  : Apply the exponential linear unit to input.
  `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

  `group_norm(...) <functional/group_norm.html>`_
  : Apply the group normalization to input.
  `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

  `l1_loss(...) <functional/l1_loss.html>`_
  : Compute the element-wise absolute value difference.

  `leaky_relu(...) <functional/leaky_relu.html>`_
  : Apply the leaky rectified linear unit to input.

  `linear(...) <functional/linear.html>`_
  : Apply the linear transformation to input.

  `local_response_norm(...) <functional/local_response_norm.html>`_
  : Apply the local response normalization to input.
  `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

  `log_softmax(...) <functional/log_softmax.html>`_
  : Apply the composite of logarithm and softmax to input.

  `interpolate(...) <functional/interpolate.html>`_
  : Resize input via interpolating neighborhoods.

  `max_pool2d(...) <functional/max_pool2d.html>`_
  : Apply the 2d max pooling to input.

  `mse_loss(...) <functional/mse_loss.html>`_
  : Compute the element-wise squared error.

  `nll_loss(...) <functional/nll_loss.html>`_
  : Compute the negative likelihood loss with sparse labels.

  `normalize(...) <functional/normalize.html>`_
  : Apply the :math:`L_{p}` normalization to the input.

  `pad(...) <functional/pad.html>`_
  : Pad the input according to the given sizes.

  `prelu(...) <functional/prelu.html>`_
  : Apply the parametric rectified linear unit to input.
  `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

  `relu(...) <functional/relu.html>`_
  : Apply rectified linear unit to input.
  `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

  `relu6(...) <functional/relu6.html>`_
  : Apply the clipped-6 rectified linear unit to input.
  `[Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_.

  `selu(...) <functional/selu.html>`_
  : Compute the sigmoid focal loss with sparse labels.
  `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`__.

  `smooth_l1_loss(...) <functional/smooth_l1_loss.html>`_
  : Compute the element-wise error transited from L1 and L2.
  `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

  `softmax(...) <functional/softmax.html>`_
  : Apply the softmax function to input.

  `sync_batch_norm(...) <functional/sync_batch_norm.html>`_
  : Apply the sync batch normalization to input.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `tanh(...) <functional/tanh.html>`_
  : Apply the tanh function to input.

  `upsample(...) <functional/upsample.html>`_
  : Upsample input via interpolating neighborhoods.

  `upsample_bilinear(...) <functional/upsample_bilinear.html>`_
  : Upsample input via bilinear interpolating.

  `upsample_nearest(...) <functional/upsample_nearest.html>`_
  : Upsample input via nearest interpolating.

.. toctree::
  :hidden:

  functional/avg_pool2d
  functional/batch_norm
  functional/binary_cross_entropy_with_logits
  functional/conv2d
  functional/conv_transpose2d
  functional/cross_entropy
  functional/ctc_loss
  functional/depthwise_conv2d
  functional/drop_block2d
  functional/drop_path
  functional/dropout
  functional/elu
  functional/group_norm
  functional/l1_loss
  functional/leaky_relu
  functional/linear
  functional/local_response_norm
  functional/log_softmax
  functional/interpolate
  functional/max_pool2d
  functional/mse_loss
  functional/nll_loss
  functional/normalize
  functional/pad
  functional/prelu
  functional/relu
  functional/relu6
  functional/selu
  functional/sigmoid
  functional/sigmoid_focal_loss
  functional/smooth_l1_loss
  functional/softmax
  functional/sync_batch_norm
  functional/tanh
  functional/upsample
  functional/upsample_bilinear
  functional/upsample_nearest

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
