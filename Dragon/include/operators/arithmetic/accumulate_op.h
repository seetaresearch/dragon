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

#ifndef DRAGON_OPERATORS_ARITHMETIC_ACCUMULATE_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_ACCUMULATE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class AccumulateOp final : public Operator<Context> {
 public:
    AccumulateOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          alpha_(OpArg<float>("alpha", 1.f)),
          beta_(OpArg<float>("beta", 1.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

    template <typename T>
    void RunImpl(Tensor* X, Tensor* Y);

 protected:
    float alpha_, beta_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_ACCUMULATE_OP_H_