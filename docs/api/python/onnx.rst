vm.onnx
=======

.. toctree::
  :hidden:

  onnx/Shell

Operators
#########

======================== ========= ========================================
Name                     Supported Reference
======================== ========= ========================================
`Abs`_                   |v|       :func:`dragon.math.abs`
`Acos`_
`Acosh`_
`Add`_                   |v|       :func:`dragon.math.add`
`And`_                   |v|       :func:`dragon.bitwise.bitwise_and`
`ArgMax`_                |v|       :func:`dragon.math.argmax`
`ArgMin`_                |v|       :func:`dragon.math.argmin`
`Asin`_
`Asinh`_
`Atan`_
`Atanh`_
`AveragePool`_           |v|       :func:`dragon.nn.pool2d`
`BatchNormalization`_    |v|       :func:`dragon.nn.batch_norm`
`BitShift`_
`Cast`_                  |v|       :func:`dragon.cast`
`Ceil`_                  |v|       :func:`dragon.math.ceil`
`Clip`_                  |v|       :func:`dragon.math.clip`
`Compress`_
`Concat`_                |v|       :func:`dragon.concat`
`ConcatFromSequence`_
`Constant`_
`ConstantOfShape`_
`Conv`_                  |v|       :func:`dragon.nn.conv2d`
`ConvInteger`_
`ConvTranspose`_         |v|       :func:`dragon.nn.conv2d_transpose`
`Cos`_                   |v|       :func:`dragon.math.cos`
`Cosh`_
`CumSum`_                |v|       :func:`dragon.math.cumsum`
`DepthToSpace`_          |v|       :func:`dragon.nn.depth_to_space`
`DequantizeLinear`_
`Det`_
`Div`_                   |v|       :func:`dragon.math.div`
`Dropout`_               |v|       :func:`dragon.nn.dropout`
`Einsum`_
`Elu`_                   |v|       :func:`dragon.nn.elu`
`Equal`_                 |v|       :func:`dragon.math.equal`
`Erf`_
`Exp`_                   |v|       :func:`dragon.math.exp`
`Expand`_                |v|       :func:`dragon.broadcast_to`
`EyeLike`_               |v|       :func:`dragon.eye_like`
`Flatten`_               |v|       :func:`dragon.flatten`
`Floor`_                 |v|       :func:`dragon.math.floor`
`GRU`_                   |v|       :func:`dragon.nn.GRU`
`Gather`_                |v|       :func:`dragon.index_select`
`GatherElements`_
`GatherND`_
`Gemm`_                  |v|       :func:`dragon.nn.fully_connected`
`GlobalAveragePool`_     |v|       :func:`dragon.nn.pool2d`
`GlobalLpPool`_
`GlobalMaxPool`_         |v|       :func:`dragon.nn.pool2d`
`Greater`_               |v|       :func:`dragon.math.greater`
`HardSigmoid`_           |v|       :func:`dragon.nn.hardsigmoid`
`Hardmax`_
`Identity`_              |v|       :func:`dragon.identity`
`If`_
`InstanceNormalization`_ |v|       :func:`dragon.nn.instance_norm`
`IsInf`_                 |v|       :func:`dragon.math.is_inf`
`IsNaN`_                 |v|       :func:`dragon.math.is_nan`
`LRN`_                   |v|       :func:`dragon.nn.lrn`
`LSTM`_                  |v|       :func:`dragon.nn.LSTM`
`LeakyRelu`_             |v|       :func:`dragon.nn.leaky_relu`
`Less`_                  |v|       :func:`dragon.math.less`
`Log`_                   |v|       :func:`dragon.math.log`
`LogSoftmax`_            |v|       :func:`dragon.nn.log_softmax`
`Loop`_
`LpNormalization`_       |v|       :func:`dragon.math.lp_normalize`
`LpPool`_
`MatMul`_                |v|       :func:`dragon.math.matmul`
`MatMulInteger`_
`Max`_                   |v|       :func:`dragon.math.maximum`
`MaxPool`_               |v|       :func:`dragon.nn.pool2d`
`MaxRoiPool`_            |v|       :func:`dragon.vision.roi_pool`
`MaxUnpool`_
`Mean`_                  |v|       :func:`dragon.math.add`
`Min`_                   |v|       :func:`dragon.math.minimum`
`Mod`_
`Mul`_                   |v|       :func:`dragon.math.mul`
`Multinomial`_           |v|       :func:`dragon.random.multinomial`
`Neg`_                   |v|       :func:`dragon.math.negative`
`NonMaxSuppression`_
`NonZero`_               |v|       :func:`dragon.nonzero`
`Not`_                   |v|       :func:`dragon.bitwise.invert`
`OneHot`_                |v|       :func:`dragon.one_hot`
`Or`_                    |v|       :func:`dragon.bitwise.bitwise_or`
`PRelu`_                 |v|       :func:`dragon.nn.prelu`
`Pad`_                   |v|       :func:`dragon.pad`
`Pow`_                   |v|       :func:`dragon.math.pow`
`QLinearConv`_
`QLinearMatMul`_
`QuantizeLinear`_
`RNN`_                   |v|       :func:`dragon.nn.RNN`
`RandomNormal`_          |v|       :func:`dragon.random.normal`
`RandomNormalLike`_      |v|       :func:`dragon.random.normal_like`
`RandomUniform`_         |v|       :func:`dragon.random.uniform`
`RandomUniformLike`_     |v|       :func:`dragon.random.uniform_like`
`Reciprocal`_            |v|       :func:`dragon.math.reciprocal`
`ReduceL1`_
`ReduceL2`_
`ReduceLogSum`_
`ReduceLogSumExp`_
`ReduceMax`_             |v|       :func:`dragon.math.max`
`ReduceMean`_            |v|       :func:`dragon.math.mean`
`ReduceMin`_             |v|       :func:`dragon.math.min`
`ReduceProd`_
`ReduceSum`_             |v|       :func:`dragon.math.sum`
`ReduceSumSquare`_
`Relu`_                  |v|       :func:`dragon.nn.relu`
`Reshape`_               |v|       :func:`dragon.reshape`
`Resize`_                |v|       :func:`dragon.vision.resize`
`ReverseSequence`_
`RoiAlign`_              |v|       :func:`dragon.vision.roi_align`
`Round`_                 |v|       :func:`dragon.math.round`
`Scan`_
`Scatter`_
`ScatterElements`_
`ScatterND`_
`Selu`_                  |v|       :func:`dragon.nn.selu`
`SequenceAt`_
`SequenceConstruct`_
`SequenceEmpty`_
`SequenceErase`_
`SequenceInsert`_
`SequenceLength`_
`Shape`_                 |v|       :func:`dragon.shape`
`Shrink`_
`Sigmoid`_               |v|       :func:`dragon.math.sigmoid`
`Sign`_                  |v|       :func:`dragon.math.sign`
`Sin`_                   |v|       :func:`dragon.math.sin`
`Sinh`_
`Size`_
`Slice`_                 |v|       :func:`dragon.slice`
`Softmax`_               |v|       :func:`dragon.nn.softmax`
`Softplus`_
`Softsign`_
`SpaceToDepth`_          |v|       :func:`dragon.nn.space_to_depth`
`Split`_                 |v|       :func:`dragon.split`
`SplitToSequence`_
`Sqrt`_                  |v|       :func:`dragon.math.sqrt`
`Squeeze`_               |v|       :func:`dragon.squeeze`
`StringNormalizer`_
`Sub`_                   |v|       :func:`dragon.math.sub`
`Sum`_                   |v|       :func:`dragon.math.add`
`Tan`_
`Tanh`_                  |v|       :func:`dragon.math.tanh`
`TfIdfVectorizer`_
`ThresholdedRelu`_
`Tile`_                  |v|       :func:`dragon.tile`
`TopK`_                  |v|       :func:`dragon.math.top_k`
`Transpose`_             |v|       :func:`dragon.transpose`
`Unique`_                |v|       :func:`dragon.unique`
`Unsqueeze`_             |v|       :func:`dragon.unsqueeze`
`Upsample`_              |v|       :func:`dragon.vision.resize`
`Where`_                 |v|       :func:`dragon.where`
`Xor`_                   |v|       :func:`dragon.bitwise.bitwise_xor`
======================== ========= ========================================

.. only:: html

    Classes
    #######

    `class Shell <onnx/Shell.html>`_
    : Context-manger to export or load onnx models.

.. _Abs: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Abs
.. _Acos: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Acos
.. _Acosh: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Acosh
.. _Add: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
.. _And: https://github.com/onnx/onnx/blob/master/docs/Operators.md#And
.. _ArgMax: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
.. _ArgMin: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin
.. _Asin: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Asin
.. _Asinh: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Asinh
.. _Atan: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Atan
.. _Atanh: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Atanh
.. _AveragePool: https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool
.. _BatchNormalization: https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
.. _BitShift: https://github.com/onnx/onnx/blob/master/docs/Operators.md#BitShift
.. _Cast: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
.. _Ceil: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
.. _Clip: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Clip
.. _Compress: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Compress
.. _Concat: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat
.. _ConcatFromSequence: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConcatFromSequence
.. _Constant: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant
.. _ConstantOfShape: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConstantOfShape
.. _Conv: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
.. _ConvInteger: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvInteger
.. _ConvTranspose: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvTranspose
.. _Cos: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cos
.. _Cosh: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cosh
.. _CumSum: https://github.com/onnx/onnx/blob/master/docs/Operators.md#CumSum
.. _DepthToSpace: https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace
.. _DequantizeLinear: https://github.com/onnx/onnx/blob/master/docs/Operators.md#DequantizeLinear
.. _Det: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Det
.. _Div: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
.. _Dropout: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout
.. _Einsum: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Einsum
.. _Elu: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Elu
.. _Equal: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
.. _Erf: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Erf
.. _Exp: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
.. _Expand: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Expand
.. _EyeLike: https://github.com/onnx/onnx/blob/master/docs/Operators.md#EyeLike
.. _Flatten: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten
.. _Floor: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Floor
.. _GRU: https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
.. _Gather: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather
.. _GatherElements: https://github.com/onnx/onnx/blob/master/docs/Operators.md#GatherElements
.. _GatherND: https://github.com/onnx/onnx/blob/master/docs/Operators.md#GatherND
.. _Gemm: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
.. _GlobalAveragePool: https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalAveragePool
.. _GlobalLpPool: https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalLpPool
.. _GlobalMaxPool: https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalMaxPool
.. _Greater: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
.. _HardSigmoid: https://github.com/onnx/onnx/blob/master/docs/Operators.md#HardSigmoid
.. _Hardmax: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Hardmax
.. _Identity: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity
.. _If: https://github.com/onnx/onnx/blob/master/docs/Operators.md#If
.. _InstanceNormalization: https://github.com/onnx/onnx/blob/master/docs/Operators.md#InstanceNormalization
.. _IsInf: https://github.com/onnx/onnx/blob/master/docs/Operators.md#IsInf
.. _IsNaN: https://github.com/onnx/onnx/blob/master/docs/Operators.md#IsNaN
.. _LRN: https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN
.. _LSTM: https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
.. _LeakyRelu: https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu
.. _Less: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
.. _Log: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
.. _LogSoftmax: https://github.com/onnx/onnx/blob/master/docs/Operators.md#LogSoftmax
.. _Loop: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop
.. _LpNormalization: https://github.com/onnx/onnx/blob/master/docs/Operators.md#LpNormalization
.. _LpPool: https://github.com/onnx/onnx/blob/master/docs/Operators.md#LpPool
.. _MatMul: https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
.. _MatMulInteger: https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMulInteger
.. _Max: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
.. _MaxPool: https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
.. _MaxRoiPool: https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxRoiPool
.. _MaxUnpool: https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxUnpool
.. _Mean: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mean
.. _Min: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
.. _Mod: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mod
.. _Mul: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul
.. _Multinomial: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Multinomial
.. _Neg: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Neg
.. _NonMaxSuppression: https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonMaxSuppression
.. _NonZero: https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonZero
.. _Not: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
.. _OneHot: https://github.com/onnx/onnx/blob/master/docs/Operators.md#OneHot
.. _Or: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Or
.. _PRelu: https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu
.. _Pad: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad
.. _Pow: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pow
.. _QLinearConv: https://github.com/onnx/onnx/blob/master/docs/Operators.md#QLinearConv
.. _QLinearMatMul: https://github.com/onnx/onnx/blob/master/docs/Operators.md#QLinearMatMul
.. _QuantizeLinear: https://github.com/onnx/onnx/blob/master/docs/Operators.md#QuantizeLinear
.. _RNN: https://github.com/onnx/onnx/blob/master/docs/Operators.md#RNN
.. _RandomNormal: https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomNormal
.. _RandomNormalLike: https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomNormalLike
.. _RandomUniform: https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomUniform
.. _RandomUniformLike: https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomUniformLike
.. _Reciprocal: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reciprocal
.. _ReduceL1: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceL1
.. _ReduceL2: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceL2
.. _ReduceLogSum: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceLogSum
.. _ReduceLogSumExp: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceLogSumExp
.. _ReduceMax: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax
.. _ReduceMean: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean
.. _ReduceMin: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin
.. _ReduceProd: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd
.. _ReduceSum: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum
.. _ReduceSumSquare: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSumSquare
.. _Relu: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
.. _Reshape: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
.. _Resize: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
.. _ReverseSequence: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReverseSequence
.. _RoiAlign: https://github.com/onnx/onnx/blob/master/docs/Operators.md#RoiAlign
.. _Round: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
.. _Scan: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scan
.. _Scatter: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scatter
.. _ScatterElements: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ScatterElements
.. _ScatterND: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ScatterND
.. _Selu: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Selu
.. _SequenceAt: https://github.com/onnx/onnx/blob/master/docs/Operators.md#SequenceAt
.. _SequenceConstruct: https://github.com/onnx/onnx/blob/master/docs/Operators.md#SequenceConstruct
.. _SequenceEmpty: https://github.com/onnx/onnx/blob/master/docs/Operators.md#SequenceEmpty
.. _SequenceErase: https://github.com/onnx/onnx/blob/master/docs/Operators.md#SequenceErase
.. _SequenceInsert: https://github.com/onnx/onnx/blob/master/docs/Operators.md#SequenceInsert
.. _SequenceLength: https://github.com/onnx/onnx/blob/master/docs/Operators.md#SequenceLength
.. _Shape: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape
.. _Shrink: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shrink
.. _Sigmoid: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid
.. _Sign: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sign
.. _Sin: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin
.. _Sinh: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sinh
.. _Size: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size
.. _Slice: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
.. _Softmax: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
.. _Softplus: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softplus
.. _Softsign: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softsign
.. _SpaceToDepth: https://github.com/onnx/onnx/blob/master/docs/Operators.md#SpaceToDepth
.. _Split: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split
.. _SplitToSequence: https://github.com/onnx/onnx/blob/master/docs/Operators.md#SplitToSequence
.. _Sqrt: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
.. _Squeeze: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
.. _StringNormalizer: https://github.com/onnx/onnx/blob/master/docs/Operators.md#StringNormalizer
.. _Sub: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
.. _Sum: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum
.. _Tan: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tan
.. _Tanh: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tanh
.. _TfIdfVectorizer: https://github.com/onnx/onnx/blob/master/docs/Operators.md#TfIdfVectorizer
.. _ThresholdedRelu: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ThresholdedRelu
.. _Tile: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tile
.. _TopK: https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
.. _Transpose: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose
.. _Unique: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unique
.. _Unsqueeze: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
.. _Upsample: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Upsample
.. _Where: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
.. _Xor: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Xor

.. |v| image:: ../_static/images/tick.png
  :height: 18

.. raw:: html

  <style>
  h1:before {
    content: "Module: dragon.";
    color: #103d3e;
  }
  </style>
