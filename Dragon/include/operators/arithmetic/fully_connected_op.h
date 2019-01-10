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

#ifndef DRAGON_OPERATORS_ARITHMETIC_FULLY_CONNECTED_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_FULLY_CONNECTED_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class FullyConnectedOp final : public Operator<Context> {
 public:
    FullyConnectedOp(const OperatorDef& def, Workspace *ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 1)),
          N(OperatorBase::Arg<int64_t>("num_output", 0)),
          transW(OperatorBase::Arg<bool>("transW", true)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice();
    template <typename T> void TransRunWithType();
    template <typename T> void NoTransRunWithType();

 protected:
    int64_t axis, transW, M, K, N;
};

template <class Context>
class FullyConnectedGradientOp final : public Operator<Context> {
 public:
    FullyConnectedGradientOp(const OperatorDef& def, Workspace *ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 1)),
          N(OperatorBase::Arg<int64_t>("num_output", 0)),
          transW(OperatorBase::Arg<bool>("transW", true)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int64_t axis, transW, M, K, N;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_FULLY_CONNECTED_OP_H_