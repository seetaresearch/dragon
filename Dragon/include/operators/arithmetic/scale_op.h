// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_SCALE_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_SCALE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ScaleOp : public Operator<Context> {
 public:
    ScaleOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)),
          num_axes(OperatorBase::GetSingleArg<int>("num_axes", -1)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, num_axes, inner_dim;
    Tensor* bias_multiplier;
};

template <class Context>
class ScaleGradientOp final : public Operator<Context> {
 public:
    ScaleGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
        axis(OperatorBase::GetSingleArg<int>("axis", 1)),
        num_axes(OperatorBase::GetSingleArg<int>("num_axes", -1)) {}

    void ShareBeforeRun() override;
    void RunOnDevice() override;
    void ClearAfterRun() override;
    template <typename T> void BiasRunWithType();
    template <typename T> void ScaleRunWithType();
    template <typename T> void RunWithType();

 protected:
    TIndex axis, num_axes;
    TIndex outer_dim, inner_dim, scale_dim, sum_dim, dim;
    Tensor* bias_multiplier, *sum_multiplier;
    Tensor sum_result;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_SCALE_OP_H_