/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_ARITHMETIC_INNER_PRODUCT_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_INNER_PRODUCT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class InnerProductOp final : public Operator<Context> {
 public:
    InnerProductOp(const OperatorDef& def, Workspace *ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 1)),
          num_output(OperatorBase::Arg<int>("num_output", 0)),
          TransW(OperatorBase::Arg<bool>("TransW", true)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice();
    template <typename T> void TransRunWithType();
    template <typename T> void NoTransRunWithType();

 protected:
    TIndex axis, num_output, TransW, M, K;
};

template <class Context>
class InnerProductGradientOp final : public Operator<Context> {
 public:
    InnerProductGradientOp(const OperatorDef& def, Workspace *ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 1)),
          num_output(OperatorBase::Arg<int>("num_output", 0)),
          TransW(OperatorBase::Arg<bool>("TransW", true)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, num_output, TransW, M, K;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_INNER_PRODUCT_OP_H_