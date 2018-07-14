// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_LOSS_SIGMOID_FOCAL_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_SIGMOID_FOCAL_LOSS_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SigmoidFocalLossOp
    final : public Operator<Context> {
 public:
    SigmoidFocalLossOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 1)),
          normalization(OperatorBase::Arg<string>(
              "normalization", "VALID")),
          alpha(OperatorBase::Arg<float>("alpha", 0.25f)),
          gamma(OperatorBase::Arg<float>("gamma", 2.f)),
          neg_id(OperatorBase::Arg<int>("neg_id", 0)) {
        pos_alpha = alpha;
        neg_alpha = 1.f - alpha;
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float alpha, gamma, pos_alpha, neg_alpha;
    TIndex axis, neg_id, outer_dim, axis_dim, inner_dim;
    Tensor losses, flags;
    string normalization;
};

template <class Context>
class SigmoidFocalLossGradientOp
    final : public Operator<Context> {
 public:
     SigmoidFocalLossGradientOp(
         const OperatorDef&         def,
         Workspace*                 ws)
         : Operator<Context>(def, ws),
           axis(OperatorBase::Arg<int>("axis", 1)),
           normalization(OperatorBase::Arg<string>(
               "normalization", "VALID")),
           alpha(OperatorBase::Arg<float>("alpha", 0.25f)),
           gamma(OperatorBase::Arg<float>("gamma", 2.f)),
           neg_id(OperatorBase::Arg<int>("neg_id", 0)) {
         pos_alpha = alpha;
         neg_alpha = 1.f - alpha;
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float alpha, gamma, pos_alpha, neg_alpha;
    TIndex axis, neg_id, outer_dim, axis_dim, inner_dim;
    Tensor flags;
    string normalization;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_LOSS_SIGMOID_FOCAL_LOSS_OP_H_