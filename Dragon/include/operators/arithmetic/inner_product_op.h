// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

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
    USE_OPERATOR_FUNCTIONS(Context);

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
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, num_output, M, K;
    bool transW;
    Tensor* bias_multiplier;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_INNER_PRODUCT_OP_H_