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
           alpha(OperatorBase::GetSingleArg<float>("alpha", 0.5)),
           gamma(OperatorBase::GetSingleArg<float>("gamma", 2.0)),
           neg_id(OperatorBase::GetSingleArg<int>("neg_id", -1)) {
         pos_alpha = alpha * 2.0;
         neg_alpha = (1 - alpha) * 2.0;
     }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float alpha, gamma; 
    int neg_id;
    float pos_alpha, neg_alpha;
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
           eps(OperatorBase::GetSingleArg<float>("eps", float(1e-10))),
           neg_id(OperatorBase::GetSingleArg<int>("neg_id", -1)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float gamma, eps;
    int neg_id;
    TIndex axis, outer_dim, inner_dim;
    Tensor* scale;
    string normalization;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_LOSS_SPARSE_SOFTMAX_FOCAL_LOSS_OP_H_