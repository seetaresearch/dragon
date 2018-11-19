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

#ifndef DRAGON_OPERATORS_ARITHMETIC_MAXIMUM_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_MAXIMUM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class MaximumOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(MaximumOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType();
};

template <class Context>
class MaximumGradientOp final : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(MaximumGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EltwiseRunWithType();
    template <typename T> void BroadcastRunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_MAXIMUM_OP_H_