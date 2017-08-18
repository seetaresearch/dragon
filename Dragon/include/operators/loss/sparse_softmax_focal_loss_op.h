// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_LOSS_SPARSE_SOFTMAX_FOCAL_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_SPARSE_SOFTMAX_FOCAL_LOSS_OP_H_

#include "operators/loss/sparse_softmax_cross_entropy_op.h"

namespace dragon {

template <class Context>
class SparseSoftmaxFocalLossOp final : public SparseSoftmaxCrossEntropyOp<Context> {
 public:
     SparseSoftmaxFocalLossOp(const OperatorDef& op_def, Workspace* ws)
         : SparseSoftmaxCrossEntropyOp<Context>(op_def, ws),
           axis(OperatorBase::GetSingleArg<int>("axis", 1)),
           normalization(OperatorBase::GetSingleArg<string>("normalization", "VALID")),
           alpha(OperatorBase::GetSingleArg<float>("alpha", 1.0)),
           gamma(OperatorBase::GetSingleArg<float>("gamma", 2.0)),
           use_pseudo_metric(OperatorBase::GetSingleArg<bool>("use_pseudo_metric", true)) {
         if (alpha == 1.0) use_pseudo_metric = false;
     }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float alpha, gamma;
    bool use_pseudo_metric;
    TIndex axis, outer_dim, inner_dim;
    Tensor* scale;
    string normalization;
};

template <class Context>
class SparseSoftmaxFocalLossGradientOp final : public SparseSoftmaxCrossEntropyGradientOp<Context> {
 public:
     SparseSoftmaxFocalLossGradientOp(const OperatorDef& op_def, Workspace* ws)
         : SparseSoftmaxCrossEntropyGradientOp<Context>(op_def, ws),
           axis(OperatorBase::GetSingleArg<int>("axis", 1)),
           normalization(OperatorBase::GetSingleArg<string>("normalization", "VALID")),
           gamma(OperatorBase::GetSingleArg<float>("gamma", 2.0)),
           eps(OperatorBase::GetSingleArg<float>("eps", float(1e-10))) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float gamma, eps;
    TIndex axis, outer_dim, inner_dim;
    Tensor* scale;
    string normalization;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_LOSS_SPARSE_SOFTMAX_FOCAL_LOSS_OP_H_