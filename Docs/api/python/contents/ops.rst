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
`LMDBData`_       Prefetch Image data with *LMDB* database.
`ImageData`_      Process the images from 4D raw data.
==============    ========================================================================

Initializer
-----------
==================   ======================================================================
List                 Brief
==================   ======================================================================
`Fill`_              Fill a Tensor with a specific value.
`RandomUniform`_     Randomly initialize a Tensor with *Uniform* distribution.
`RandomNormal`_      Randomly initialize a Tensor with *Normal* distribution.
`TruncatedNormal`_   Randomly initialize a Tensor with *Truncated Normal* distribution.
`GlorotUniform`_     Randomly initialize a Tensor with *Xavier Uniform* distribution.
`GlorotNormal`_      Randomly initialize a Tensor with *Kaiming Normal* distribution.
==================   ======================================================================

Vision
------
===================    ======================================================================
List                   Brief
===================    ======================================================================
`Conv2d`_              2d Convolution.
`DepthwiseConv2d`_     Depthwise 2d Convolution. `[Chollet, 2016] <https://arxiv.org/abs/1610.02357>`_.
`Conv2dTranspose`_     2d Deconvolution.
`Pool2d`_              2d Pooling, *MAX* or *AVG*.
`ROIPool`_             RoI Pooling (*MAX*). `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.
`ROIAlign`_            RoI Align (*AVG*). `[He et.al, 2017] <https://arxiv.org/abs/1703.06870>`_.
`LRN`_                 Local Response Normalization. `[Krizhevsky et.al, 2012] <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>`_.
`NNResize`_            Resize the image with *Nearest-Neighbor* method.
`BilinearResize`_      Resize the image with *Bi-Linear* method.
`BiasAdd`_             Add the bias across channels to a *NCHW* or *NHWC* input.
`DropBlock2d`_         Randomly drop the outputs according to the spatial blocks. `[Ghiasi et.al, 2018] <https://arxiv.org/abs/1810.12890>`_.
===================    ======================================================================

Recurrent
---------
===============    ======================================================================
List               Brief
===============    ======================================================================
`RNN`_             Multi-layer *Elman-RNN* with *TanH* or *ReLU* non-linearity. `[Elman, 1990] <https://doi.org/10.1016>`_.
`LSTM`_            Multi-layer Long Short-Term Memory(*LSTM*) RNN. `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.
`GRU`_             Multi-layer Gated Recurrent Unit (*GRU*) RNN. `[Cho et.al, 2014] <https://arxiv.org/abs/1406.1078>`_.
`LSTMCell`_        Single-layer Long Short-Term Memory(*LSTM*) Cell. `[Hochreiter & Schmidhuber, 1997] <https://doi.org/10.1162>`_.
===============    ======================================================================

Activation
----------
===============    ======================================================================
List               Brief
===============    ======================================================================
`Sigmoid`_         *Sigmoid* function.
`Tanh`_            *TanH* function.
`Relu`_            Rectified Linear Unit function. `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.
`LRelu`_           Leaky Rectified Linear Unit function.
`PRelu`_           Parametric Rectified Linear Unit function. `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.
`Elu`_             Exponential Linear Unit function. `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.
`SElu`_            Scaled Exponential Linear Unit function. `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.
`Softmax`_         *Softmax* function.
`Dropout`_         Randomly set a unit into zero. `[Srivastava et.al, 2014] <http://jmlr.org/papers/v15/srivastava14a.html>`_.
`DropPath`_        Randomly set a example of batch into zero. `[Larsson et.al, 2016] <https://arxiv.org/abs/1605.07648>`_.
===============    ======================================================================

Loss
----
=============================     =====================================================================================
List                              Brief
=============================     =====================================================================================
`NLLLoss`_                        Compute the negative likelihood loss with sparse labels.
`SparseSoftmaxCrossEntropy`_      Compute the softmax cross entropy with sparse labels.
`SigmoidCrossEntropy`_            Compute sigmoid cross entropy with given logits and targets.
`SoftmaxCrossEntropy`_            Compute the softmax cross entropy with given logits and one-hot labels.
`SmoothL1Loss`_                   Compute the smoothed L1 loss. `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.
`L1Loss`_                         Compute the L1 loss.
`L2Loss`_                         Compute the L2 loss.
`SigmoidFocalLoss`_               Compute the sigmoid focal loss with sparse labels. `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.
`SoftmaxFocalLoss`_               Compute the softmax focal loss with sparse labels. `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`_.
`CTCLoss`_                        Compute the ctc loss with batched variable length of labels. `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.
=============================     =====================================================================================

Arithmetic
----------
==================    ======================================================================
List                  Brief
==================    ======================================================================
`Add`_                Calculate *A + B*.
`Sub`_                Calculate *A - B*.
`Mul`_                Calculate *A * B*.
`Div`_                Calculate *A / B*.
`Dot`_                Calculate the vector dot.
`Pow`_                Calculate the power of input.
`Log`_                Calculate the logarithm of input.
`Exp`_                Calculate the exponential of input.
`Square`_             Calculate the square of input.
`Sqrt`_               Calculate the sqrt of input.
`Maximum`_            Return the max value of given two inputs.
`Minimum`_            Return the min value of given two inputs.
`Clip`_               Clip the input to be between lower and higher bounds.
`Matmul`_             Matrix Multiplication.
`FullyConnected`_     Calculate *Y = X * W' + b*.
`Eltwise`_            Element-wise Sum or Product the arbitrary number of inputs.
`Affine`_             Calculate *Y = Ax + b* along the given range of axes.
`GramMatrix`_         Calculate the gram matrix. `[Gatys et.al, 2016] <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf>`_.
`Moments`_            Calculate the mean and variance of inputs along the given axes.
`Accumulate`_         Calculate *y = alpha * x + beta * y*
`MovingAverage`_      Calculate the *y = (1 - decay) * x + decay * y*
==================    ======================================================================

Normalization
-------------
==================    ======================================================================
List                  Brief
==================    ======================================================================
`BatchNorm`_          Batch Normalization. `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.
`GroupNorm`_          Group Normalization. `[Wu & He, 2018] <https://arxiv.org/abs/1803.08494>`_.
`LayerNorm`_          Layer Normalization. `[Ba et.al, 2016] <https://arxiv.org/abs/1607.06450>`_
`InstanceNorm`_       Instance Normalization. `[Ulyanov et.al, 2016] <https://arxiv.org/abs/1607.08022>`_.
`L2Norm`_             L2 Normalization. `[Liu et.al, 2015] <https://arxiv.org/abs/1506.04579>`_.
==================    ======================================================================

Array
-----
==================    ======================================================================
List                  Brief
==================    ======================================================================
`Where`_              Select elements from either *x* or *y*.
`IndexSelect`_        Select the elements according to the indices along the given axis.
`MaskedSelect`_       Select the the elements where *mask* is *1*.
`Reduce`_             Reduce the inputs along the axis in given axes.
`Sum`_                Compute the sum along the given axis.
`Mean`_               Compute the mean along the given axis.
`Max`_                Compute the values of maximum elements along the given axis.
`ArgMax`_             Compute the indices of maximum elements along the given axis.
`Min`_                Compute the values of minimum elements along the given axis.
`ArgMin`_             Compute the indices of minimum elements along the given axis.
`Slice`_              Slice the inputs into several parts along the given axis.
`Stack`_              Stack the inputs along the given axis.
`Concat`_             Concatenate the inputs along the given axis.
`ChannelShuffle`_     Shuffle channels between groups along the given axis. `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.
`Repeat`_             Repeat the input along the given axis.
`Transpose`_          Transpose the input according to the given permutations.
`Tile`_               Tile the input according to the given multiples.
`Pad`_                Pad the input according to the given sizes.
`Crop`_               Crop the input according to the given starts and sizes.
`OneHot`_             Generate the one-hot representation of inputs.
`Flatten`_            Flatten the input along the given axes.
`Reshape`_            Reshape the dimensions of input.
`Squeeze`_            Remove the dimensions with size 1.
`ExpandDims`_         Expand the new dimension with size 1 to specific axis.
`Shape`_              Get the dynamic shape of a Tensor.
`NonZero`_            Return the indices of non-zero elements.
`Arange`_             Return evenly spaced values within a given interval.
`Multinomial`_        Return indices sampled from the multinomial distribution.
==================    ======================================================================

Control Flow
------------
===============    ======================================================================
List               Brief
===============    ======================================================================
`Copy`_            Copy the *value* to *ref*.
`Assign`_          Assign the *value* to *ref*.
`MaskedAssign`_    Assign the *value* to *ref* where mask is *1*.
`Equal`_           *Equal* Comparing between A and B.
`NotEqual`_        *NotEqual* Comparing between A and B.
`Less`_            *Less* Comparing between A and B.
`LessEqual`_       *LessEqual* Comparing between A and B.
`Greater`_         *Greater* Comparing between A and B.
`GreaterEqual`_    *GreaterEqual* Comparing between A and B.
===============    ======================================================================

Misc
----
=================    ======================================================================
List                 Brief
=================    ======================================================================
`Cast`_              Cast the data type of inputs to a specific one.
`Run`_               Run a custom operator. (Without GradientFlow)
`Template`_          Run a custom operator. (With GradientFlow)
`Accuracy`_          Calculate the Top-K accuracy.
`StopGradient`_      Return the identity of input with truncated gradient flow.
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
`MPIBroadcast`_      Broadcast a tensor to all nodes in the *MPIGroup*.
`MPIGather`_         Gather a tensor from all nodes to root in the *MPIGroup*.
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
.. _DepthwiseConv2d: operators/vision.html#dragon.operators.vision.DepthwiseConv2d
.. _Conv2dTranspose: operators/vision.html#dragon.operators.vision.Conv2dTranspose
.. _Pool2d: operators/vision.html#dragon.operators.vision.Pool2d
.. _ROIPool: operators/vision.html#dragon.operators.vision.ROIPool
.. _ROIAlign: operators/vision.html#dragon.operators.vision.ROIAlign
.. _LRN: operators/vision.html#dragon.operators.vision.LRN
.. _NNResize: operators/vision.html#dragon.operators.vision.NNResize
.. _BilinearResize: operators/vision.html#dragon.operators.vision.BilinearResize
.. _BiasAdd: operators/vision.html#dragon.operators.vision.BiasAdd
.. _DenseConcat: operators/vision.html#dragon.operators.vision.DenseConcat
.. _DropBlock2d: operators/vision.html#dragon.operators.vision.DropBlock2d

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
.. _DropPath: operators/activation.html#dragon.operators.activation.DropPath

.. _NLLLoss: operators/loss.html#dragon.operators.loss.NLLLoss
.. _SparseSoftmaxCrossEntropy: operators/loss.html#dragon.operators.loss.SparseSoftmaxCrossEntropy
.. _SigmoidCrossEntropy: operators/loss.html#dragon.operators.loss.SigmoidCrossEntropy
.. _SoftmaxCrossEntropy: operators/loss.html#dragon.operators.loss.SoftmaxCrossEntropy
.. _SmoothL1Loss: operators/loss.html#dragon.operators.loss.SmoothL1Loss
.. _L1Loss: operators/loss.html#dragon.operators.loss.L1Loss
.. _L2Loss: operators/loss.html#dragon.operators.loss.L2Loss
.. _SigmoidFocalLoss: operators/loss.html#dragon.operators.loss.SigmoidFocalLoss
.. _SoftmaxFocalLoss: operators/loss.html#dragon.operators.loss.SoftmaxFocalLoss
.. _CTCLoss: operators/loss.html#dragon.operators.loss.CTCLoss

.. _Add: operators/arithmetic.html#dragon.operators.arithmetic.Add
.. _Sub: operators/arithmetic.html#dragon.operators.arithmetic.Sub
.. _Mul: operators/arithmetic.html#dragon.operators.arithmetic.Mul
.. _Div: operators/arithmetic.html#dragon.operators.arithmetic.Div
.. _Clip: operators/arithmetic.html#dragon.operators.arithmetic.Clip
.. _Maximum: operators/arithmetic.html#dragon.operators.arithmetic.Maximum
.. _Minimum: operators/arithmetic.html#dragon.operators.arithmetic.Minimum
.. _Pow: operators/arithmetic.html#dragon.operators.arithmetic.Pow
.. _Log: operators/arithmetic.html#dragon.operators.arithmetic.Log
.. _Exp: operators/arithmetic.html#dragon.operators.arithmetic.Exp
.. _Square: operators/arithmetic.html#dragon.operators.arithmetic.Square
.. _Sqrt: operators/arithmetic.html#dragon.operators.arithmetic.Square
.. _Matmul: operators/arithmetic.html#dragon.operators.arithmetic.Matmul
.. _Dot: operators/arithmetic.html#dragon.operators.arithmetic.Dot
.. _FullyConnected: operators/arithmetic.html#dragon.operators.arithmetic.FullyConnected
.. _Eltwise: operators/arithmetic.html#dragon.operators.arithmetic.Eltwise
.. _Affine: operators/arithmetic.html#dragon.operators.arithmetic.Affine
.. _GramMatrix: operators/arithmetic.html#dragon.operators.arithmetic.GramMatrix
.. _Moments: operators/arithmetic.html#dragon.operators.arithmetic.Moments
.. _Accumulate: operators/arithmetic.html#dragon.operators.arithmetic.Accumulate
.. _MovingAverage: operators/arithmetic.html#dragon.operators.arithmetic.MovingAverage

.. _BatchNorm: operators/norm.html#dragon.operators.norm.BatchNorm
.. _GroupNorm: operators/norm.html#dragon.operators.norm.GroupNorm
.. _LayerNorm: operators/norm.html#dragon.operators.norm.LayerNorm
.. _InstanceNorm: operators/norm.html#dragon.operators.norm.InstanceNorm
.. _L2Norm: operators/norm.html#dragon.operators.norm.L2Norm

.. _Where: operators/array.html#dragon.operators.array.Where
.. _IndexSelect: operators/array.html#dragon.operators.array.IndexSelect
.. _MaskedSelect: operators/array.html#dragon.operators.array.MaskedSelect
.. _Crop: operators/array.html#dragon.operators.array.Crop
.. _Reduce: operators/array.html#dragon.operators.array.Reduce
.. _Sum: operators/array.html#dragon.operators.array.Sum
.. _Mean: operators/array.html#dragon.operators.array.Mean
.. _Max: operators/array.html#dragon.operators.array.Max
.. _ArgMax: operators/array.html#dragon.operators.array.ArgMax
.. _Min: operators/array.html#dragon.operators.array.Min
.. _ArgMin: operators/array.html#dragon.operators.array.ArgMin
.. _Slice: operators/array.html#dragon.operators.array.Slice
.. _Stack: operators/array.html#dragon.operators.array.Stack
.. _Concat: operators/array.html#dragon.operators.array.Concat
.. _ChannelShuffle: operators/array.html#dragon.operators.array.ChannelShuffle
.. _Transpose: operators/array.html#dragon.operators.array.Transpose
.. _Repeat: operators/array.html#dragon.operators.array.Repeat
.. _Tile: operators/array.html#dragon.operators.array.Tile
.. _Pad: operators/array.html#dragon.operators.array.Pad
.. _OneHot: operators/array.html#dragon.operators.array.OneHot
.. _Flatten: operators/array.html#dragon.operators.array.Flatten
.. _Reshape: operators/array.html#dragon.operators.array.Reshape
.. _Squeeze: operators/array.html#dragon.operators.array.Squeeze
.. _ExpandDims: operators/array.html#dragon.operators.array.ExpandDims
.. _Shape: operators/array.html#dragon.operators.array.Shape
.. _Arange: operators/array.html#dragon.operators.array.Arange
.. _NonZero: operators/array.html#dragon.operators.array.NonZero
.. _Multinomial: operators/array.html#dragon.operators.array.Multinomial

.. _Copy: operators/control_flow.html#dragon.operators.control_flow.Copy
.. _Assign: operators/control_flow.html#dragon.operators.control_flow.Assign
.. _MaskedAssign: operators/control_flow.html#dragon.operators.control_flow.MaskedAssign
.. _Equal: operators/control_flow.html#dragon.operators.control_flow.Equal
.. _NotEqual: operators/control_flow.html#dragon.operators.control_flow.NotEqual
.. _Less: operators/control_flow.html#dragon.operators.control_flow.Less
.. _LessEqual: operators/control_flow.html#dragon.operators.control_flow.LessEqual
.. _Greater: operators/control_flow.html#dragon.operators.control_flow.Greater
.. _GreaterEqual: operators/control_flow.html#dragon.operators.control_flow.GreaterEqual

.. _Cast: operators/misc.html#dragon.operators.misc.Cast
.. _Run: operators/misc.html#dragon.operators.misc.Run
.. _Template: operators/misc.html#dragon.operators.misc.Template
.. _Accuracy: operators/misc.html#dragon.operators.misc.Accuracy
.. _StopGradient: operators/misc.html#dragon.operators.misc.StopGradient

.. _Proposal: operators/contrib/rcnn.html#dragon.operators.contrib.rcnn.ops.Proposal

.. _MPIBroadcast: operators/mpi.html#dragon.operators.mpi.MPIBroadcast
.. _MPIGather: operators/mpi.html#dragon.operators.mpi.MPIGather