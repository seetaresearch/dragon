// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_INNER_PRODUCT_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_INNER_PRODUCT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class InnerProductOp: public Operator<Context> {
 public:
    InnerProductOp(const OperatorDef& op_def, Workspace *ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)),
          num_output(OperatorBase::GetSingleArg<int>("num_output", 0)),
          transW(OperatorBase::GetSingleArg<bool>("TransW", true)) {}

    void RunOnDevice();
    template <typename T> void TransRunWithType();
    template <typename T> void NoTransRunWithType();

 protected:
    TIndex axis, num_output, M, K;
    bool transW;
    Tensor* bias_multiplier;
};

template <class Context>
class InnerProductGradientOp final : public Operator<Context> {
 public:
    InnerProductGradientOp(const OperatorDef& op_def, Workspace *ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 1)),
          num_output(OperatorBase::GetSingleArg<int>("num_output", 0)),
          transW(OperatorBase::GetSingleArg<bool>("TransW", true)) {}

    void ShareBeforeRun() override;
    void RunOnDevice() override;
    void ClearAfterRun() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, num_output, M, K;
    bool transW;
    Tensor* bias_multiplier;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_INNER_PRODUCT_OP_H_