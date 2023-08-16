vm.torch.nn.functional
======================

.. only:: html

  Functions
  ---------

  `adaptive_avg_pool1d(...) <functional/adaptive_avg_pool1d.html>`_
  : Apply 1d adaptive average pooling to input.

  `adaptive_avg_pool2d(...) <functional/adaptive_avg_pool2d.html>`_
  : Apply 2d adaptive average pooling to input.

  `adaptive_avg_pool3d(...) <functional/adaptive_avg_pool3d.html>`_
  : Apply 3d adaptive average pooling to input.

  `adaptive_max_pool1d(...) <functional/adaptive_max_pool1d.html>`_
  : Apply 1d adaptive max pooling to input.

  `adaptive_max_pool2d(...) <functional/adaptive_max_pool2d.html>`_
  : Apply 2d adaptive max pooling to input.

  `adaptive_max_pool3d(...) <functional/adaptive_max_pool3d.html>`_
  : Apply 3d adaptive max pooling to input.

  `affine(...) <functional/affine.html>`_
  : Apply affine transformation to input.

  `avg_pool1d(...) <functional/avg_pool1d.html>`_
  : Apply 1d average pooling to input.

  `avg_pool2d(...) <functional/avg_pool2d.html>`_
  : Apply 2d average pooling to input.

  `avg_pool3d(...) <functional/avg_pool3d.html>`_
  : Apply 3d average pooling to input.

  `batch_norm(...) <functional/batch_norm.html>`_
  : Apply batch normalization to input.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `binary_cross_entropy_with_logits(...) <functional/binary_cross_entropy_with_logits.html>`_
  : Compute sigmoid cross entropy.

  `channel_norm(...) <nn/channel_norm.html>`_
  : Apply normalization to each channel of input.

  `channel_shuffle(...) <functional/channel_shuffle.html>`_
  : Apply group shuffle to each channel of input.
  `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

  `conv1d(...) <functional/conv1d.html>`_
  : Apply 1d convolution to input.

  `conv2d(...) <functional/conv2d.html>`_
  : Apply 2d convolution to input.

  `conv3d(...) <functional/conv3d.html>`_
  : Apply 3d convolution to input.

  `conv_transpose1d(...) <functional/conv_transpose1d.html>`_
  : Apply 1d deconvolution to input.

  `conv_transpose2d(...) <functional/conv_transpose2d.html>`_
  : Apply 2d deconvolution to input.

  `conv_transpose3d(...) <functional/conv_transpose3d.html>`_
  : Apply 3d deconvolution to input.

  `cosine_similarity(...) <functional/cosine_similarity.html>`_
  : Compute cosine similarity between inputs.

  `cross_entropy(...) <functional/cross_entropy.html>`_
  : Compute softmax cross entropy.

  `ctc_loss(...) <functional/ctc_loss.html>`_
  : Compute ctc loss.
  `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

  `drop_block2d(...) <functional/drop_block2d.html>`_
  : Set blocks of input to zero randomly.
  `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.

  `drop_path(...) <functional/drop_path.html>`_
  : Set examples of input to zero randomly.
  `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.

  `dropout(...) <functional/dropout.html>`_
  : Set elements of input to zero randomly.
  `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

  `elu(...) <functional/elu.html>`_
  : Apply exponential linear unit to input.
  `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

  `gelu(...) <functional/gelu.html>`_
  : Apply gaussian error linear unit to input.
  `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

  `group_norm(...) <functional/group_norm.html>`_
  : Apply group normalization to input.
  `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.

  `hardsigmoid(...) <functional/hardsigmoid.html>`_
  : Apply hard sigmoid function to input.

  `hardswish(...) <functional/hardswish.html>`_
  : Apply hard swish function to input.
  `[Howard et.al, 2019] <https://arxiv.org/abs/1905.02244>`_.

  `kl_div(...) <functional/kl_div.html>`_
  : Compute Kullback-Leibler divergence.

  `l1_loss(...) <functional/l1_loss.html>`_
  : Compute element-wise absolute value difference.

  `layer_norm(...) <functional/layer_norm.html>`_
  : Apply layer normalization to input.
  `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_

  `leaky_relu(...) <functional/leaky_relu.html>`_
  : Apply leaky rectified linear unit to input.

  `linear(...) <functional/linear.html>`_
  : Apply linear transformation to input.

  `local_response_norm(...) <functional/local_response_norm.html>`_
  : Apply local response normalization to input.
  `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

  `log_softmax(...) <functional/log_softmax.html>`_
  : Apply logarithm softmax function to input.

  `interpolate(...) <functional/interpolate.html>`_
  : Resize input via interpolating neighborhoods.

  `max_pool1d(...) <functional/max_pool1d.html>`_
  : Apply 1d max pooling to input.

  `max_pool2d(...) <functional/max_pool2d.html>`_
  : Apply 2d max pooling to input.

  `max_pool3d(...) <functional/max_pool3d.html>`_
  : Apply 3d max pooling to input.

  `mse_loss(...) <functional/mse_loss.html>`_
  : Compute element-wise squared error.

  `multi_head_attention_forward(...) <functional/multi_head_attention_forward.html>`_
  : Apply multihead attention to input.
  `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

  `nll_loss(...) <functional/nll_loss.html>`_
  : Compute negative likelihood loss.

  `normalize(...) <functional/normalize.html>`_
  : Apply :math:`L_{p}` normalization to the input.

  `one_hot(...) <functional/one_hot.html>`_
  : Return one-hot representation of input.

  `pad(...) <functional/pad.html>`_
  : Pad input according to the given sizes.

  `pixel_shuffle(...) <functional/pixel_shuffle.html>`_
  : Rearrange depth elements of input into pixels.

  `pixel_unshuffle(...) <functional/pixel_unshuffle.html>`_
  : Rearrange pixels of input into depth elements.

  `prelu(...) <functional/prelu.html>`_
  : Apply parametric rectified linear unit to input.
  `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

  `relu(...) <functional/relu.html>`_
  : Apply rectified linear unit to input.
  `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

  `relu6(...) <functional/relu6.html>`_
  : Apply clipped-6 rectified linear unit to input.
  `[Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_.

  `selu(...) <functional/selu.html>`_
  : Compute sigmoid focal loss.
  `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`__.

  `sigmoid(...) <functional/sigmoid.html>`_
  : Apply sigmoid function to input.

  `sigmoid_focal_loss(...) <functional/sigmoid_focal_loss.html>`_
  : Compute sigmoid focal loss.
  `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`__.

  `silu(...) <functional/silu.html>`_
  : Apply sigmoid linear unit to input.
  `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

  `smooth_l1_loss(...) <functional/smooth_l1_loss.html>`_
  : Compute element-wise error transited from L1 and L2.
  `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

  `softmax(...) <functional/softmax.html>`_
  : Apply softmax function to input.

  `sync_batch_norm(...) <functional/sync_batch_norm.html>`_
  : Apply sync batch normalization to input.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `tanh(...) <functional/tanh.html>`_
  : Apply tanh function to input.

  `unfold(...) <functional/unfold.html>`_
  : Extract sliding blocks from input.

  `upsample(...) <functional/upsample.html>`_
  : Upsample input via interpolating neighborhoods.

  `upsample_bilinear(...) <functional/upsample_bilinear.html>`_
  : Upsample input via bilinear interpolating.

  `upsample_nearest(...) <functional/upsample_nearest.html>`_
  : Upsample input via nearest interpolating.

.. toctree::
  :hidden:

  functional/adaptive_avg_pool1d
  functional/adaptive_avg_pool2d
  functional/adaptive_avg_pool3d
  functional/adaptive_max_pool1d
  functional/adaptive_max_pool2d
  functional/adaptive_max_pool3d
  functional/affine
  functional/avg_pool1d
  functional/avg_pool2d
  functional/avg_pool3d
  functional/batch_norm
  functional/binary_cross_entropy_with_logits
  functional/channel_norm
  functional/channel_shuffle
  functional/conv1d
  functional/conv2d
  functional/conv3d
  functional/conv_transpose1d
  functional/conv_transpose2d
  functional/conv_transpose3d
  functional/cosine_similarity
  functional/cross_entropy
  functional/ctc_loss
  functional/drop_block2d
  functional/drop_path
  functional/dropout
  functional/elu
  functional/gelu
  functional/group_norm
  functional/hardsigmoid
  functional/hardswish
  functional/kl_div
  functional/l1_loss
  functional/leaky_relu
  functional/linear
  functional/layer_norm
  functional/local_response_norm
  functional/log_softmax
  functional/interpolate
  functional/max_pool1d
  functional/max_pool2d
  functional/max_pool3d
  functional/mse_loss
  functional/multi_head_attention_forward
  functional/nll_loss
  functional/normalize
  functional/one_hot
  functional/pad
  functional/pixel_shuffle
  functional/pixel_unshuffle
  functional/prelu
  functional/relu
  functional/relu6
  functional/selu
  functional/sigmoid
  functional/sigmoid_focal_loss
  functional/silu
  functional/smooth_l1_loss
  functional/softmax
  functional/sync_batch_norm
  functional/tanh
  functional/unfold
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
