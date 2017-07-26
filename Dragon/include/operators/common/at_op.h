// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_COMMON_AT_OP_H_
#define DRAGON_OPERATORS_COMMON_AT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class AtOp final : public Operator<Context> {
 public:
    AtOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 0)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    vector<TIndex> output_dims;
};

template <class Context>
class AtGradientOp final : public Operator<Context> {
 public:
    AtGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 0)),
          acc_grad(OperatorBase::GetSingleArg<bool>("acc_gradient", false)) {}

    void ShareBeforeRun() override;
    void RunOnDevice() override;
    void ClearAfterRun() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, outer_dim, inner_dim, x_slice_dim, y_slice_dim;
    bool acc_grad;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_COMMON_AT_OP_H_