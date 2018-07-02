=================
:mod:`dragon.ops`
=================

.. toctree::
   :hidden:

Data
----
==============    ========================================================================
List              Brief
==============    ========================================================================
`LMDBData`_       Prefetch Image data with LMDB database.
`ImageData`_      Process the images from 4D raw data.
==============    ========================================================================

Initializer
-----------
==================   ======================================================================
List                 Brief
==================   ======================================================================
`Fill`_              Fill a Tensor with a specific value.
`RandomUniform`_     Randomly initialize a Tensor with uniform distribution.
`RandomNormal`_      Randomly initialize a Tensor with normal distribution.
`TruncatedNormal`_   Randomly initialize a Tensor with truncated normal distribution.
`GlorotUniform`_     Randomly initialize a Tensor with Xavier uniform distribution.
`GlorotNormal`_      Randomly initialize a Tensor with MSRA normal distribution.
==================   ======================================================================

Vision
------
===================    ======================================================================
List                   Brief
===================    ======================================================================
`Conv2d`_              2d Convolution.
`Conv2dTranspose`_     2d Deconvolution.
`Pool2d`_              2d Pooling, MAX or AVG.
`ROIPooling`_          ROIPooling(MAX). `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.
`ROIAlign`_            ROIAlign(AVG). `[He et.al, 2017] <https://arxiv.org/abs/1703.06870>`_.
`LRN`_                 Local Response Normalization. `[Krizhevsky et.al, 2012] <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>`_.
`NNResize`_            Resize the image with Nearest-Neighbor method.
`BilinearResize`_      Resize the image with Bi-linear method.
`BiasAdd`_             Add the bias across channels to a ``NCHW`` or ``NHWC`` input.
`DenseConcat`_         Memory-efficient concatenation for DenseNet. `[Huang et.al, 2017] <http://arxiv.org/abs/1608.06993>`_.
===================    ======================================================================

Recurrent
---------
===============    ======================================================================
List               Brief
===============    ======================================================================
`RNN`_             Multi-layer Elman-RNN with `TanH` or `ReLU` non-linearity. `[Elman, 1990] <https://doi.org/10.1016>`_.
`LSTM`_            Multi-layer Long Short-Term Memory(LSTM) RNN. `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.
`GRU`_             Multi-layer Gated Recurrent Unit (GRU) RNN. `[Cho et.al, 2014] <https://arxiv.org/abs/1406.1078>`_.
`LSTMCell`_        Single-layer Long Short-Term Memory(LSTM) Cell. `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.
===============    ======================================================================

Activation
----------
===============    ======================================================================
List               Brief
===============    ======================================================================
`Sigmoid`_         Sigmoid function.
`Tanh`_            Tanh function.
`Relu`_            Rectified Linear Unit function. `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.
`LRelu`_           Leaky Rectified Linear Unit function.
`PRelu`_           Parametric Rectified Linear Unit function. `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.
`Elu`_             Exponential Linear Unit function. `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.
`SElu`_            Scaled Exponential Linear Unit function. `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.
`Softmax`_         Softmax function.
`Dropout`_         Randomly set a unit into zero. `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.
===============    ======================================================================

Loss
----
=============================      ======================================================================
List                               Brief
=============================      ======================================================================
`SparseSoftmaxCrossEntropy`_       SoftmaxCrossEntropy with sparse labels.
`SigmoidCrossEntropy`_             SigmoidCrossEntropy.
`SoftmaxCrossEntropy`_             SoftmaxCrossEntropy with dense(one-hot) labels.
`SmoothL1Loss`_                    SmoothL1Loss. `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.
`L1Loss`_                          L1Loss.
`L2Loss`_                          L2Loss(EuclideanLoss).
`SparseSoftmaxFocalLoss`_          SoftmaxFocalLoss with sparse labels. `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.
`CTCLoss`_                         CTCLoss with batched variable length of labels. `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.
=============================      ======================================================================

Arithmetic
----------
===============    ======================================================================
List               Brief
===============    ======================================================================
`Add`_             Calculate A + B.
`Sub`_             Calculate A - B.
`Mul`_             Calculate A * B.
`Div`_             Calculate A / B.
`Dot`_             Calculate A dot B.
`Pow`_             Calculate the power of input.
`Log`_             Calculate the logarithm of input.
`Exp`_             Calculate the exponential of input.
`Square`_          Calculate the square of input.
`Sqrt`_            Calculate the sqrt of input.
`Clip`_            Clip the input to be between lower and higher bounds.
`Matmul`_          Matrix Multiplication.
`InnerProduct`_    InnerProduct Function.
`Eltwise`_         Eltwise Sum/Product Function.
`Affine`_          Calculate ``y = Ax + b`` along the given range of axes.
`GramMatrix`_      Calculate the gram matrix. `[Gatys et.al, 2016] <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf>`_.
===============    ======================================================================

Normalization
-------------
==================    ======================================================================
List                  Brief
==================    ======================================================================
`BatchNorm`_          Batch Normalization. `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.
`BatchRenorm`_        Batch Renormalization. `[Ioffe, 2017] <https://arxiv.org/abs/1702.03275>`_.
`FusedBatchNorm`_     Batch Normalization, with scale procedure after normalization.
`GroupNorm`_          Group Normalization. `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.
`FusedGroupNorm`_     Group Normalization, with scale procedure after normalization.
`InstanceNorm`_       Instance Normalization. `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_.
`L2Norm`_             L2 Normalization. `[Liu et.al, 2015] <https://arxiv.org/abs/1506.04579>`_.
==================    ======================================================================

NDArray
-------
===============    ======================================================================
List               Brief
===============    ======================================================================
`Gather`_          Gather the input according to the indices along the given axis.
`RandomPick`_      Randomly pick the input along the given axis.
`Reduce`_          The general reduce operator.
`Sum`_             Compute the sum along the given axis.
`Mean`_            Compute the mean along the given axis.
`Max`_             Compute the values of maximum elements along the given axis.
`Argmax`_          Compute the indices of maximum elements along the given axis.
`Min`_             Compute the values of minimum elements along the given axis.
`Argmin`_          Compute the indices of minimum elements along the given axis.
`Slice`_           Slice interface of NDArray.
`Stack`_           Stack the inputs along the given axis.
`Concat`_          Concatenate the inputs along the given axis.
`Repeat`_          Repeat the input along the given axis.
`Transpose`_       Transpose the input according to the given permutations.
`Tile`_            Tile the input according to the given multiples.
`Pad`_             Pad the input according to the given paddings.
`Crop`_            Crop the input according to the given starts and ends.
`OneHot`_          Generate the one-hot representation of inputs.
`Flatten`_         Flatten the input along the given axes.
`Reshape`_         Reshape the dimensions of input.
`ExpandDims`_      ExpandDims interface of NDArray.
`Shape`_           Get the dynamic shape of a Tensor.
`Arange`_          Return a vector of elements by arange.
===============    ======================================================================

Control Flow
------------
===============    ======================================================================
List               Brief
===============    ======================================================================
`Copy`_            Copy A to B.
`Equal`_           Equal Comparing between A and B.
===============    ======================================================================

Misc
----
=================    ======================================================================
List                 Brief
=================    ======================================================================
`AsType`_            Cast the data type of inputs to a specific one.
`Run`_               Run a custom operator. (Without GradientFlow)
`Template`_          Run a custom operator. (With GradientFlow)
`Accuracy`_          Calculate the Top-K accuracy.
`StopGradient`_      Return the identity of input with truncated gradient flow.
`MovingAverage`_     Calculate the moving average.
=================    ======================================================================

Contrib
-------
=================    ======================================================================
List                 Brief
=================    ======================================================================
`Proposal`_          Generate Regional Proposals. `[Ren et.al, 2015] <https://arxiv.org/abs/1506.01497>`_.
=================    ======================================================================

MPI
---
=================    ======================================================================
List                 Brief
=================    ======================================================================
`MPIBroadcast`_      Broadcast a tensor to all nodes in the ``MPIGroup``.
`MPIGather`_         Gather a tensor from all nodes to root in the ``MPIGroup``.
=================    ======================================================================


.. _LMDBData: operators/data.html#dragon.operators.data.LMDBData
.. _ImageData: operators/data.html#dragon.operators.data.ImageData

.. _Fill: operators/initializer.html#dragon.operators.initializer.Fill
.. _RandomUniform: operators/initializer.html#dragon.operators.initializer.RandomUniform
.. _RandomNormal: operators/initializer.html#dragon.operators.initializer.RandomNormal
.. _TruncatedNormal: operators/initializer.html#dragon.operators.initializer.TruncatedNormal
.. _GlorotUniform: operators/initializer.html#dragon.operators.initializer.GlorotUniform
.. _GlorotNormal: operators/initializer.html#dragon.operators.initializer.GlorotNormal

.. _Conv2d: operators/vision.html#dragon.operators.vision.Conv2d
.. _Conv2dTranspose: operators/vision.html#dragon.operators.vision.Conv2dTranspose
.. _Pool2d: operators/vision.html#dragon.operators.vision.Pool2d
.. _ROIPooling: operators/vision.html#dragon.operators.vision.ROIPooling
.. _ROIAlign: operators/vision.html#dragon.operators.vision.ROIAlign
.. _LRN: operators/vision.html#dragon.operators.vision.LRN
.. _NNResize: operators/vision.html#dragon.operators.vision.NNResize
.. _BilinearResize: operators/vision.html#dragon.operators.vision.BilinearResize
.. _BiasAdd: operators/vision.html#dragon.operators.vision.BiasAdd
.. _DenseConcat: operators/vision.html#dragon.operators.vision.DenseConcat

.. _RNN: operators/recurrent.html#dragon.operators.recurrent.RNN
.. _LSTM: operators/recurrent.html#dragon.operators.recurrent.LSTM
.. _GRU: operators/recurrent.html#dragon.operators.recurrent.GRU
.. _LSTMCell: operators/recurrent.html#dragon.operators.recurrent.LSTMCell

.. _Sigmoid: operators/activation.html#dragon.operators.activation.Sigmoid
.. _Tanh: operators/activation.html#dragon.operators.activation.Tanh
.. _Relu: operators/activation.html#dragon.operators.activation.Relu
.. _LRelu: operators/activation.html#dragon.operators.activation.LRelu
.. _PRelu: operators/activation.html#dragon.operators.activation.PRelu
.. _Elu: operators/activation.html#dragon.operators.activation.Elu
.. _SElu: operators/activation.html#dragon.operators.activation.SElu
.. _Softmax: operators/activation.html#dragon.operators.activation.Softmax
.. _Dropout: operators/activation.html#dragon.operators.activation.Dropout

.. _SparseSoftmaxCrossEntropy: operators/loss.html#dragon.operators.loss.SparseSoftmaxCrossEntropy
.. _SigmoidCrossEntropy: operators/loss.html#dragon.operators.loss.SigmoidCrossEntropy
.. _SoftmaxCrossEntropy: operators/loss.html#dragon.operators.loss.SoftmaxCrossEntropy
.. _SmoothL1Loss: operators/loss.html#dragon.operators.loss.SmoothL1Loss
.. _L1Loss: operators/loss.html#dragon.operators.loss.L1Loss
.. _L2Loss: operators/loss.html#dragon.operators.loss.L2Loss
.. _SparseSoftmaxFocalLoss: operators/loss.html#dragon.operators.loss.SparseSoftmaxFocalLoss
.. _CTCLoss: operators/loss.html#dragon.operators.loss.CTCLoss

.. _Add: operators/arithmetic.html#dragon.operators.arithmetic.Add
.. _Sub: operators/arithmetic.html#dragon.operators.arithmetic.Sub
.. _Mul: operators/arithmetic.html#dragon.operators.arithmetic.Mul
.. _Div: operators/arithmetic.html#dragon.operators.arithmetic.Div
.. _Clip: operators/arithmetic.html#dragon.operators.arithmetic.Clip
.. _Pow: operators/arithmetic.html#dragon.operators.arithmetic.Pow
.. _Log: operators/arithmetic.html#dragon.operators.arithmetic.Log
.. _Exp: operators/arithmetic.html#dragon.operators.arithmetic.Exp
.. _Square: operators/arithmetic.html#dragon.operators.arithmetic.Square
.. _Sqrt: operators/arithmetic.html#dragon.operators.arithmetic.Square
.. _Matmul: operators/arithmetic.html#dragon.operators.arithmetic.Matmul
.. _Dot: operators/arithmetic.html#dragon.operators.arithmetic.Dot
.. _InnerProduct: operators/arithmetic.html#dragon.operators.arithmetic.InnerProduct
.. _Eltwise: operators/arithmetic.html#dragon.operators.arithmetic.Eltwise
.. _Affine: operators/arithmetic.html#dragon.operators.arithmetic.Affine
.. _GramMatrix: operators/arithmetic.html#dragon.operators.arithmetic.GramMatrix

.. _BatchNorm: operators/norm.html#dragon.operators.norm.BatchNorm
.. _BatchRenorm: operators/norm.html#dragon.operators.norm.BatchRenorm
.. _FusedBatchNorm: operators/norm.html#dragon.operators.norm.FusedBatchNorm
.. _GroupNorm: operators/norm.html#dragon.operators.norm.GroupNorm
.. _FusedGroupNorm: operators/norm.html#dragon.operators.norm.FusedGroupNorm
.. _InstanceNorm: operators/norm.html#dragon.operators.norm.InstanceNorm
.. _L2Norm: operators/norm.html#dragon.operators.norm.L2Norm

.. _Gather: operators/ndarray.html#dragon.operators.ndarray.Gather
.. _RandomPick: operators/ndarray.html#dragon.operators.ndarray.RandomPick
.. _Crop: operators/ndarray.html#dragon.operators.ndarray.Crop
.. _Reduce: operators/ndarray.html#dragon.operators.ndarray.Reduce
.. _Sum: operators/ndarray.html#dragon.operators.ndarray.Sum
.. _Mean: operators/ndarray.html#dragon.operators.ndarray.Mean
.. _Max: operators/ndarray.html#dragon.operators.ndarray.Max
.. _Argmax: operators/ndarray.html#dragon.operators.ndarray.Argmax
.. _Min: operators/ndarray.html#dragon.operators.ndarray.Min
.. _Argmin: operators/ndarray.html#dragon.operators.ndarray.Argmin
.. _Slice: operators/ndarray.html#dragon.operators.ndarray.Slice
.. _Stack: operators/ndarray.html#dragon.operators.ndarray.Stack
.. _Concat: operators/ndarray.html#dragon.operators.ndarray.Concat
.. _Transpose: operators/ndarray.html#dragon.operators.ndarray.Transpose
.. _Repeat: operators/ndarray.html#dragon.operators.ndarray.Repeat
.. _Tile: operators/ndarray.html#dragon.operators.ndarray.Tile
.. _Pad: operators/ndarray.html#dragon.operators.ndarray.Pad
.. _OneHot: operators/ndarray.html#dragon.operators.ndarray.OneHot
.. _Flatten: operators/ndarray.html#dragon.operators.ndarray.Flatten
.. _Reshape: operators/ndarray.html#dragon.operators.ndarray.Reshape
.. _ExpandDims: operators/ndarray.html#dragon.operators.ndarray.ExpandDims
.. _Shape: operators/ndarray.html#dragon.operators.ndarray.Shape
.. _Arange: operators/ndarray.html#dragon.operators.ndarray.Arange

.. _Copy: operators/control_flow.html#dragon.operators.control_flow.Copy
.. _Equal: operators/control_flow.html#dragon.operators.control_flow.Equal

.. _AsType: operators/misc.html#dragon.operators.misc.AsType
.. _Run: operators/misc.html#dragon.operators.misc.Run
.. _Template: operators/misc.html#dragon.operators.misc.Template
.. _Accuracy: operators/misc.html#dragon.operators.misc.Accuracy
.. _StopGradient: operators/misc.html#dragon.operators.misc.StopGradient
.. _MovingAverage: operators/misc.html#dragon.operators.misc.MovingAverage

.. _Proposal: operators/contrib/rcnn.html#dragon.operators.contrib.rcnn.ops.Proposal

.. _MPIBroadcast: operators/mpi.html#dragon.operators.mpi.MPIBroadcast
.. _MPIGather: operators/mpi.html#dragon.operators.mpi.MPIGather