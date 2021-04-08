vm.caffe.layers
===============

.. only:: html

  Classes
  -------

  `class Accuracy <layers/Accuracy.html>`_
  : Compute the top-k accuracy.

  `class ArgMax <layers/ArgMax.html>`_
  : Compute the index of maximum elements along the given axis.

  `class BatchNorm <layers/BatchNorm.html>`_
  : Apply the batch normalization.
  `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

  `class Concat <layers/Concat.html>`_
  : Concatenate the inputs along the given axis.

  `class Convolution <layers/Convolution.html>`_
  : Apply the n-dimension convolution.

  `class Crop <layers/Crop.html>`_
  : Select the elements according to the dimensions of second bottom.

  `class Data <layers/Data.html>`_
  : Load batch of data for image classification.

  `class Deconvolution <layers/Deconvolution.html>`_
  : Apply the n-dimension deconvolution.

  `class Dropout <layers/Dropout.html>`_
  : Set the elements of the input to zero randomly.
  `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.

  `class Eltwise <layers/Eltwise.html>`_
  : Compute the element-wise operation on the sequence of inputs.

  `class ELU <layers/ELU.html>`_
  : Apply the exponential linear unit.
  `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

  `class EuclideanLoss <layers/EuclideanLoss.html>`_
  : Compute the element-wise squared error.

  `class Flatten <layers/Flatten.html>`_
  : Flatten the input along the given axes.

  `class InnerProduct <layers/InnerProduct.html>`_
  : Compute the dense matrix multiplication along the given axes.

  `class Input <layers/Input.html>`_
  : Produce input blobs with shape and dtype.

  `class LRN <layers/LRN.html>`_
  : Apply the local response normalization.
  `[Krizhevsky et.al, 2012] <http://www.cs.toronto.edu/~hinton/absps/imagenet.pdf>`_.

  `class Normalize <layers/Normalize.html>`_
  : Apply the fused L2 normalization.
  `[Liu et.al, 2015] <https://arxiv.org/abs/1506.04579>`_.

  `class Permute <layers/Permute.html>`_
  : Permute the dimensions of input.

  `class Pooling <layers/Pooling.html>`_
  : Apply the n-dimension pooling.

  `class Power <layers/Power.html>`_
  : Compute the power of input.

  `class PReLU <layers/PReLU.html>`_
  : Apply the parametric rectified linear unit.
  `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

  `class Python <layers/Python.html>`_
  : Wrap a python class into a layer.

  `class Reduction <layers/Reduction.html>`_
  : Compute the reduction value along the given axis.

  `class ReLU <layers/ReLU.html>`_
  : Apply the rectified linear unit.
  `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

  `class Reshape <layers/Reshape.html>`_
  : Change the dimensions of input.

  `class Scale <layers/Scale.html>`_
  : Compute the affine transformation along the given axes.

  `class Sigmoid <layers/Sigmoid.html>`_
  : Apply the sigmoid function.

  `class SigmoidCrossEntropyLoss <layers/SigmoidCrossEntropyLoss.html>`_
  : Compute the sigmoid cross entropy with contiguous targets.

  `class SmoothL1Loss <layers/SmoothL1Loss.html>`_
  : Compute the element-wise error transited from L1 and L2.
  `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

  `class Softmax <layers/Softmax.html>`_
  : Apply the softmax function.

  `class SoftmaxWithLoss <layers/SoftmaxWithLoss.html>`_
  : Compute the softmax cross entropy with sparse labels.

  `class TanH <layers/TanH.html>`_
  : Apply the tanh function.

  `class Tile <layers/Tile.html>`_
  : Repeat the input according to the given axis.

.. toctree::
  :hidden:

  layers/Accuracy
  layers/ArgMax
  layers/BatchNorm
  layers/Concat
  layers/Convolution
  layers/Crop
  layers/Data
  layers/Deconvolution
  layers/Dropout
  layers/Eltwise
  layers/ELU
  layers/EuclideanLoss
  layers/Flatten
  layers/InnerProduct
  layers/Input
  layers/LRN
  layers/Normalize
  layers/Permute
  layers/Pooling
  layers/Power
  layers/PReLU
  layers/Python
  layers/Reduction
  layers/ReLU
  layers/Reshape
  layers/Scale
  layers/Sigmoid
  layers/SigmoidCrossEntropyLoss
  layers/SmoothL1Loss
  layers/Softmax
  layers/SoftmaxWithLoss
  layers/StopGradient
  layers/TanH
  layers/Tile

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
