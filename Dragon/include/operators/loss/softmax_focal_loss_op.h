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

#ifndef DRAGON_OPERATORS_LOSS_SOFTMAX_FOCAL_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_SOFTMAX_FOCAL_LOSS_OP_H_

#include "operators/loss/sparse_softmax_ce_loss_op.h"

namespace dragon {

template <class Context>
class SoftmaxFocalLossOp final :
    public SparseSoftmaxCrossEntropyOp<Context> {
 public:
    SoftmaxFocalLossOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : SparseSoftmaxCrossEntropyOp<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)),
          neg_id_(OpArg<int64_t>("neg_id", 0)),
          alpha_(OpArg<float>("alpha", 0.25f)),
          gamma_(OpArg<float>("gamma", 2.f)),
          reduction_(OpArg<string>(
              "reduction", "VALID")) {
        pos_alpha_ = alpha_;
        neg_alpha_ = 1.f - alpha_;
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunImpl();

 protected:
    string reduction_;
    Tensor loss_, flag_;
    float alpha_, gamma_, pos_alpha_, neg_alpha_;
    int64_t axis_, neg_id_, outer_dim_, inner_dim_;
};

template <class Context>
class SoftmaxFocalLossGradientOp final
    : public SparseSoftmaxCrossEntropyGradientOp<Context> {
 public:
    SoftmaxFocalLossGradientOp(
        const OperatorDef&          def,
        Workspace*                  ws)
         : SparseSoftmaxCrossEntropyGradientOp<Context>(def, ws),
           axis_(OpArg<int64_t>("axis", 1)),
           neg_id_(OpArg<int64_t>("neg_id", 0)),
           alpha_(OpArg<float>("alpha", 0.25f)),
           gamma_(OpArg<float>("gamma", 2.f)),
           reduction_(OpArg<string>("reduction", "VALID")) {
        pos_alpha_ = alpha_;
        neg_alpha_ = 1.f - alpha_;
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunImpl();

 protected:
    Tensor flag_;
    string reduction_;
    float alpha_, gamma_, pos_alpha_, neg_alpha_;
    int64_t axis_, neg_id_, outer_dim_, inner_dim_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_LOSS_SOFTMAX_FOCAL_LOSS_OP_H_