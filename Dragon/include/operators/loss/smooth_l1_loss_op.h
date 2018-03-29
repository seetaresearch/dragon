// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_LOSS_SMOOTH_L1_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_SMOOTH_L1_LOSS_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SmoothL1LossOp final : public Operator<Context> {
 public:
    SmoothL1LossOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          sigma2(OperatorBase::GetSingleArg<float>("sigma", 1.0)),
          normalization(OperatorBase::GetSingleArg<string>("normalization", "BATCH_SIZE")) {
        sigma2 *= sigma2; 
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float sigma2;
    Tensor* diff, *error;
    string normalization;
};    

template <class Context>
class SmoothL1LossGradientOp final : public Operator<Context> {
 public:
    SmoothL1LossGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
        sigma2(OperatorBase::GetSingleArg<float>("sigma", 1.0)),
        normalization(OperatorBase::GetSingleArg<string>("normalization", "BATCH_SIZE")) {
        sigma2 *= sigma2;
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float sigma2;
    Tensor* diff;
    string normalization;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_LOSS_SMOOTH_L1_LOSS_OP_H_