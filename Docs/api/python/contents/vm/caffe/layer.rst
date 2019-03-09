============
:mod:`Layer`
============

.. toctree::
   :hidden:

Table of Layers
---------------

Data
####

====================    =============================================================================
List                    Brief
====================    =============================================================================
`DataLayer`_            The implementation of ``DataLayer``.
`MemoryDataLayer`_      The implementation of ``MemoryDataLayer``.
====================    =============================================================================

Vision
######

==============================      =============================================================================
List                                Brief
==============================      =============================================================================
`ConvolutionLayer`_                 The implementation of ``ConvolutionLayer``.
`DepthwiseConvolutionLayer`_        The implementation of ``DepthwiseConvolutionLayer``.
`DeconvolutionLayer`_               The implementation of ``DeconvolutionLayer``.
`PoolingLayer`_                     The implementation of ``PoolingLayer``.
`ROIPoolingLayer`_                  The implementation of ``ROIPoolingLayer``.
`ROIAlignLayer`_                    The implementation of ``ROIAlignLayer``.
`LRNLayer`_                         The implementation of ``LRNLayer``.
`NNResizeLayer`_                    The implementation of ``NNResizeLayer``.
`BilinearResizeLayer`_              The implementation of ``BilinearResizeLayer``.
`DropBlockLayer`_                   The implementation of ``DropBlockLayer``.
==============================      =============================================================================


Neuron
######

====================    =============================================================================
List                    Brief
====================    =============================================================================
`ReLULayer`_            The implementation of ``ReLULayer``.
`PReLULayer`_           The implementation of ``PReLULayer``.
`ELULayer`_             The implementation of ``ELULayer``.
`SELULayer`_            The implementation of ``SELULayer``.
`SigmoidLayer`_         The implementation of ``SigmoidLayer``.
`TanHLayer`_            The implementation of ``TanHLayer``.
`DropoutLayer`_         The implementation of ``DropoutLayer``.
`PowerLayer`_           The implementation of ``PowerLayer``.
====================    =============================================================================

Common
######

========================    =============================================================================
List                        Brief
========================    =============================================================================
`InnerProductLayer`_        The implementation of ``InnerProductLayer``.
`AccuracyLayer`_            The implementation of ``AccuracyLayer``.
`PythonLayer`_              The implementation of ``PythonLayer``.
`EltwiseLayer`_             The implementation of ``EltwiseLayer``
`AddLayer`_                 The extended implementation of ``EltwiseLayer``.
`ConcatLayer`_              The implementation of ``ConcatLayer``.
`SliceLayer`_               The implementation of ``SliceLayer``.
`CropLayer`_                The implementation of ``CropLayer``.
`ReshapeLayer`_             The implementation of ``ReshapeLayer``.
`PermuteLayer`_             The implementation of ``PermuteLayer``.
`FlattenLayer`_             The implementation of ``FlattenLayer``.
`GatherLayer`_              The extended implementation for ``GatherOp``.
`SoftmaxLayer`_             The implementation of ``SoftmaxLayer``.
`ArgMaxLayer`_              The implementation of ``ArgMaxLayer``.
`BatchNormLayer`_           The implementation of ``BatchNormLayer``.
`GroupNormLayer`_           The implementation of ``GroupNormLayer``.
`InstanceNormLayer`_        The implementation of ``InstanceNormLayer``.
`ScaleLayer`_               The implementation of ``ScaleLayer``.
`BNLayer`_                  The implementation of ``BNLayer``.
`GNLayer`_                  The implementation of ``GNLayer``.
`NormalizeLayer`_           The implementation of ``NormalizeLayer``.
`TileLayer`_                The extended implementation of ``TileLayer``.
`ExpandDimsLayer`_          The implementation of ``ExpandDimsLayer``.
`StopGradientLayer`_        The implementation of ``StopGradientLayer``.
`ProposalLayer`_            The implementation of ``ProposalLayer``.
========================    =============================================================================

Loss
####

=================================     =============================================================================
List                                  Brief
=================================     =============================================================================
`SoftmaxWithLossLayer`_               The implementation of ``SoftmaxWithLossLayer``.
`SigmoidCrossEntropyLossLayer`_       The implementation of ``SigmoidCrossEntropyLossLayer``.
`L2LossLayer`_                        The implementation of ``L2LossLayer``.
`SmoothL1LossLayer`_                  The implementation of ``SmoothL1LossLayer``.
`SigmoidWithFocalLossLayer`_          The implementation of ``SigmoidWithFocalLossLayer``.
`SoftmaxWithFocalLossLayer`_          The implementation of ``SoftmaxWithFocalLossLayer``.
=================================     =============================================================================

MPI
###

=================================     =============================================================================
List                                  Brief
=================================     =============================================================================
`MPIBroadcastLayer`_                  The implementation of ``MPIBroadcastLayer``
`MPIGatherLayer`_                     The implementation of ``MPIGatherLayer``
=================================     =============================================================================


Quick Reference
---------------

====================    =============================================================================
List                    Brief
====================    =============================================================================
`Layer.Setup`_          Setup the parameters.
`Layer.Fill`_           Register the fillers.
====================    =============================================================================

API Reference
-------------

.. currentmodule:: dragon.vm.caffe.layer

.. autoclass:: Layer
    :members:

    .. automethod:: __init__

.. automodule:: dragon.vm.caffe.layers.data
    :members:

.. automodule:: dragon.vm.caffe.layers.vision
    :members:

.. automodule:: dragon.vm.caffe.layers.neuron
    :members:

.. automodule:: dragon.vm.caffe.layers.common
    :members:

.. automodule:: dragon.vm.caffe.layers.loss
    :members:

.. automodule:: dragon.vm.caffe.layers.mpi
    :members:


.. _DataLayer: #dragon.vm.caffe.layers.data.DataLayer
.. _MemoryDataLayer: #dragon.vm.caffe.layers.data.MemoryDataLayer

.. _ConvolutionLayer: #dragon.vm.caffe.layers.vision.ConvolutionLayer
.. _DepthwiseConvolutionLayer: #dragon.vm.caffe.layers.vision.DepthwiseConvolutionLayer
.. _DeconvolutionLayer: #dragon.vm.caffe.layers.vision.DeconvolutionLayer
.. _PoolingLayer: #dragon.vm.caffe.layers.vision.PoolingLayer
.. _ROIPoolingLayer: #dragon.vm.caffe.layers.vision.ROIPoolingLayer
.. _ROIAlignLayer: #dragon.vm.caffe.layers.vision.ROIAlignLayer
.. _LRNLayer: #dragon.vm.caffe.layers.vision.LRNLayer
.. _NNResizeLayer: #dragon.vm.caffe.layers.vision.NNResizeLayer
.. _BilinearResizeLayer: #dragon.vm.caffe.layers.vision.BilinearResizeLayer
.. _DropBlockLayer: #dragon.vm.caffe.layers.vision.DropBlockLayer

.. _ReLULayer: #dragon.vm.caffe.layers.neuron.ReLULayer
.. _PReLULayer: #dragon.vm.caffe.layers.neuron.PReLULayer
.. _ELULayer: #dragon.vm.caffe.layers.neuron.ELULayer
.. _SELULayer: #dragon.vm.caffe.layers.neuron.SELULayer
.. _SigmoidLayer: #dragon.vm.caffe.layers.neuron.SigmoidLayer
.. _TanHLayer: #dragon.vm.caffe.layers.neuron.TanHLayer
.. _DropoutLayer: #dragon.vm.caffe.layers.neuron.DropoutLayer
.. _PowerLayer: #dragon.vm.caffe.layers.neuron.PowerLayer
.. _InnerProductLayer: #dragon.vm.caffe.layers.common.InnerProductLayer
.. _AccuracyLayer: #dragon.vm.caffe.layers.common.AccuracyLayer
.. _PythonLayer: #dragon.vm.caffe.layers.common.PythonLayer
.. _EltwiseLayer: #dragon.vm.caffe.layers.common.EltwiseLayer
.. _AddLayer: #dragon.vm.caffe.layers.common.AddLayer
.. _ConcatLayer: #dragon.vm.caffe.layers.common.ConcatLayer
.. _SliceLayer: #dragon.vm.caffe.layers.common.SliceLayer
.. _CropLayer: #dragon.vm.caffe.layers.common.CropLayer
.. _ReshapeLayer: #dragon.vm.caffe.layers.common.ReshapeLayer
.. _PermuteLayer: #dragon.vm.caffe.layers.common.PermuteLayer
.. _FlattenLayer: #dragon.vm.caffe.layers.common.FlattenLayer
.. _GatherLayer: #dragon.vm.caffe.layers.common.GatherLayer
.. _SoftmaxLayer: #dragon.vm.caffe.layers.common.SoftmaxLayer
.. _ArgMaxLayer: #dragon.vm.caffe.layers.common.ArgMaxLayer
.. _BatchNormLayer: #dragon.vm.caffe.layers.common.BatchNormLayer
.. _GroupNormLayer: #dragon.vm.caffe.layers.common.GroupNormLayer
.. _InstanceNormLayer: #dragon.vm.caffe.layers.common.InstanceNormLayer
.. _ScaleLayer: #dragon.vm.caffe.layers.common.ScaleLayer
.. _BNLayer: #dragon.vm.caffe.layers.common.BNLayer
.. _GNLayer: #dragon.vm.caffe.layers.common.GNLayer
.. _NormalizeLayer: #dragon.vm.caffe.layers.common.NormalizeLayer
.. _TileLayer: #dragon.vm.caffe.layers.common.TileLayer
.. _ExpandDimsLayer: #dragon.vm.caffe.layers.common.ExpandDimsLayer
.. _StopGradientLayer: #dragon.vm.caffe.layers.common.StopGradientLayer
.. _ProposalLayer: #dragon.vm.caffe.layers.common.ProposalLayer

.. _SoftmaxWithLossLayer: #dragon.vm.caffe.layers.loss.SoftmaxWithLossLayer
.. _SigmoidCrossEntropyLossLayer: #dragon.vm.caffe.layers.loss.SigmoidCrossEntropyLossLayer
.. _L2LossLayer: #dragon.vm.caffe.layers.loss.L2LossLayer
.. _SmoothL1LossLayer: #dragon.vm.caffe.layers.loss.SmoothL1LossLayer
.. _SigmoidWithFocalLossLayer: #dragon.vm.caffe.layers.loss.SigmoidWithFocalLossLayer
.. _SoftmaxWithFocalLossLayer: #dragon.vm.caffe.layers.loss.SoftmaxWithFocalLossLayer

.. _MPIBroadcastLayer: #dragon.vm.caffe.layers.mpi.MPIBroadcastLayer
.. _MPIGatherLayer: #dragon.vm.caffe.layers.mpi.MPIGatherLayer

.. _Layer.Setup: #dragon.vm.caffe.layer.Layer.Setup
.. _Layer.Fill: #dragon.vm.caffe.layer.Layer.Fill

.. _LMDB: http://lmdb.readthedocs.io/en/release
.. _LayerSetUp(layer.hpp, L91): https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/include/caffe/layer.hpp#L91
.. _DataParameter.source: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L647
.. _DataParameter.prefetch: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L672
.. _DataParameter.batch_size: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L649
.. _ConvolutionParameter.num_output: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L571
.. _ConvolutionParameter.bias_term: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L572
.. _ConvolutionParameter.pad: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L576
.. _ConvolutionParameter.kernel_size: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L577
.. _ConvolutionParameter.stride: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L578
.. _ConvolutionParameter.dilation: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L582
.. _ConvolutionParameter.group: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L593
.. _ConvolutionParameter.weight_filler: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L595
.. _ConvolutionParameter.bias_filler: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L596
.. _PoolingParameter.pool: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L903
.. _PoolingParameter.pad: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L906
.. _PoolingParameter.pad_h: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L907
.. _PoolingParameter.pad_w: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L908
.. _PoolingParameter.kernel_size: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L909
.. _PoolingParameter.kernel_h: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L910
.. _PoolingParameter.kernel_w: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L911
.. _PoolingParameter.stride: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L912
.. _PoolingParameter.stride_h: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L913
.. _PoolingParameter.stride_w: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L914
.. _ROIPoolingParameter.pooled_h: https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/proto/caffe.proto#L1004
.. _ROIPoolingParameter.pooled_w: https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/proto/caffe.proto#L1005
.. _ROIPoolingParameter.spatial_scale: https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/proto/caffe.proto#L1008
.. _LRNParameter.local_size: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L858
.. _LRNParameter.alpha: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L859
.. _LRNParameter.beta: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L860
.. _LRNParameter.norm_region: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L865
.. _LRNParameter.k: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L866
.. _ResizeParameter.shape: https://github.com/neopenx/Dragon/tree/master/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1464
.. _ResizeParameter.fx: https://github.com/neopenx/Dragon/tree/master/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1465
.. _ResizeParameter.fy: https://github.com/neopenx/Dragon/tree/master/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1466

.. _ReLUParameter.negative_slope: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1000
.. _PReLUParameter.filler: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1409
.. _PReLUParameter.channel_shared: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1411
.. _ELUParameter.alpha: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L717
.. _DropoutParameter.dropout_ratio: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L676
.. _DropoutParameter.scale_train: https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/proto/caffe.proto#L638
.. _PowerParameter.power: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L928
.. _PowerParameter.scale: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L929
.. _PowerParameter.shift: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L930
.. _InnerProductParameter.num_output: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L822
.. _InnerProductParameter.bias_term: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L823
.. _InnerProductParameter.weight_filler: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L824
.. _InnerProductParameter.bias_filler: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L825
.. _InnerProductParameter.axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L830
.. _InnerProductParameter.transpose: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L835
.. _AccuracyParameter.top_k: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L469
.. _AccuracyParameter.axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L476
.. _AccuracyParameter.ignore_label: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L479
.. _PythonParameter.module: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L934
.. _PythonParameter.layer: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L935
.. _PythonParameter.param_str: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L940
.. _EltwiseParameter.operation: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L704
.. _EltwiseParameter.coeff: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L705
.. _ConcatParameter.axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L498
.. _CropParameter.axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L637
.. _CropParameter.offset: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L638
.. _ReshapeParameter.shape: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1030
.. _PermuteParameter.order: https://github.com/weiliu89/caffe/blob/f5eac041aafbc8b86954bd161710f65e70042ce6/src/caffe/proto/caffe.proto#L1347
.. _FlattenParameter.axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L748
.. _FlattenParameter.end_axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L753
.. _SoftmaxParameter.axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1142
.. _ArgMaxParameter.top_k: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L485
.. _ArgMaxParameter.axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L490
.. _BatchNormParameter.use_global_stats: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L511
.. _BatchNormParameter.moving_average_fraction: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L520
.. _BatchNormParameter.eps: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L523
.. _ScaleParameter.axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1087
.. _ScaleParameter.num_axes: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1095
.. _ScaleParameter.filler: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1102
.. _ScaleParameter.bias_term: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1106
.. _ScaleParameter.bias_filler: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L1107
.. _NormalizeParameter.across_spatial: https://github.com/weiliu89/caffe/blob/f5eac041aafbc8b86954bd161710f65e70042ce6/src/caffe/proto/caffe.proto#L1330
.. _NormalizeParameter.scale_filler: https://github.com/weiliu89/caffe/blob/f5eac041aafbc8b86954bd161710f65e70042ce6/src/caffe/proto/caffe.proto#L1332
.. _NormalizeParameter.channel_shared: https://github.com/weiliu89/caffe/blob/f5eac041aafbc8b86954bd161710f65e70042ce6/src/caffe/proto/caffe.proto#L1334
.. _NormalizeParameter.eps: https://github.com/weiliu89/caffe/blob/f5eac041aafbc8b86954bd161710f65e70042ce6/src/caffe/proto/caffe.proto#L1336
.. _ReductionParameter.operation: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L973
.. _ReductionParameter.axis: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L988
.. _TileParameter.multiples: https://github.com/neopenx/Dragon/blob/6eeac5fec58ed3d0d79f0b4003471e4a641c72f4/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1173
.. _ExpandDimsParameter.axis: https://github.com/neopenx/Dragon/blob/6eeac5fec58ed3d0d79f0b4003471e4a641c72f4/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1480
.. _ProposalParameter.feat_stride: https://github.com/sanghoon/caffe/blob/6068dd04ea93cca9fcee036628fdb3ea95b4ebcd/src/caffe/proto/caffe.proto#L431
.. _ProposalParameter.base_size: https://github.com/sanghoon/caffe/blob/6068dd04ea93cca9fcee036628fdb3ea95b4ebcd/src/caffe/proto/caffe.proto#L432
.. _ProposalParameter.min_size: https://github.com/sanghoon/caffe/blob/6068dd04ea93cca9fcee036628fdb3ea95b4ebcd/src/caffe/proto/caffe.proto#L433
.. _ProposalParameter.ratio: https://github.com/sanghoon/caffe/blob/6068dd04ea93cca9fcee036628fdb3ea95b4ebcd/src/caffe/proto/caffe.proto#L434
.. _ProposalParameter.scale: https://github.com/sanghoon/caffe/blob/6068dd04ea93cca9fcee036628fdb3ea95b4ebcd/src/caffe/proto/caffe.proto#L435
.. _ProposalParameter.pre_nms_topn: https://github.com/sanghoon/caffe/blob/6068dd04ea93cca9fcee036628fdb3ea95b4ebcd/src/caffe/proto/caffe.proto#L436
.. _ProposalParameter.post_nms_topn: https://github.com/sanghoon/caffe/blob/6068dd04ea93cca9fcee036628fdb3ea95b4ebcd/src/caffe/proto/caffe.proto#L437
.. _ProposalParameter.nms_thresh: https://github.com/sanghoon/caffe/blob/6068dd04ea93cca9fcee036628fdb3ea95b4ebcd/src/caffe/proto/caffe.proto#L438

.. _LossParameter.ignore_label: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L436
.. _LossParameter.normalization: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L455
.. _LossParameter.normalize: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L459
.. _SmoothL1LossParameter.sigma: https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/proto/caffe.proto#L1061
.. _FocalLossParameter.alpha: https://github.com/neopenx/Dragon/blob/6eeac5fec58ed3d0d79f0b4003471e4a641c72f4/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1509
.. _FocalLossParameter.gamma: https://github.com/neopenx/Dragon/blob/6eeac5fec58ed3d0d79f0b4003471e4a641c72f4/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1510
.. _FocalLossParameter.eps: https://github.com/neopenx/Dragon/blob/6eeac5fec58ed3d0d79f0b4003471e4a641c72f4/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1511
.. _FocalLossParameter.neg_id: https://github.com/neopenx/Dragon/blob/6eeac5fec58ed3d0d79f0b4003471e4a641c72f4/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1512

.. _MPIParameter.root: https://github.com/neopenx/Dragon/blob/6eeac5fec58ed3d0d79f0b4003471e4a641c72f4/Dragon/python/dragon/vm/caffe/proto/caffe.proto#L1445

.. _LayerParameter.phase: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L319
.. _TransformationParameter.scale: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L416
.. _TransformationParameter.mirror: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L418
.. _TransformationParameter.crop_size: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L420
.. _TransformationParameter.mean_value: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L426
.. _TransformationParameter.force_color: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L428
.. _TransformationParameter: https://github.com/BVLC/caffe/blob/effcdb0b62410b2a6a54f18f23cf90733a115673/src/caffe/proto/caffe.proto#L412
