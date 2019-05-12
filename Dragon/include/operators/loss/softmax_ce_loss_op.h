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

#ifndef DRAGON_OPERATORS_LOSS_SOFTMAX_CROSS_ENTROPY_OP_H_
#define DRAGON_OPERATORS_LOSS_SOFTMAX_CROSS_ENTROPY_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SoftmaxCrossEntropyOp final
    : public Operator<Context> {
 public:
    SoftmaxCrossEntropyOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)),
          reduction_(OpArg<string>(
              "reduction", "MEAN")) {}
    USE_OPERATOR_FUNCTIONS;

    void SoftmaxRun();
    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    Tensor loss_;
    string reduction_;
    int64_t axis_, outer_dim_, inner_dim_;
    unique_ptr<OperatorBase> softmax_op_;
};

template <class Context>
class SoftmaxCrossEntropyGradientOp final
    : public Operator<Context> {
 public:
    SoftmaxCrossEntropyGradientOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)),
          reduction_(OpArg<string>(
              "reduction", "MEAN")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    string reduction_;
    int64_t axis_, outer_dim_, inner_dim_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_LOSS_SOFTMAX_CROSS_ENTROPY_OP_H_