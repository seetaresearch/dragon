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

#ifndef DRAGON_OPERATORS_VISION_DROP_BLOCK_OP_H_
#define DRAGON_OPERATORS_VISION_DROP_BLOCK_OP_H_

#include "core/operator.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
class DropBlock2dOp final : public Operator<Context> {
 public:
    DropBlock2dOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          block_size(OperatorBase::Arg<int>("block_size", 7)),
          alpha(OperatorBase::Arg<float>("alpha", 1.f)),
          decrement(OperatorBase::Arg<float>("decrement", 0.f)),
          data_format(OperatorBase::Arg<string>("data_format", "NCHW")) {
        GET_ARGUMENT_WITH_DESC(float, keep_prob, 0.9f);
        SwitchToPhase(OperatorBase::Arg<string>("phase", ""));
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    DECLARE_ARGUMENT_WITH_DESC(float, keep_prob);
    TIndex block_size, seed_h, seed_w;
    TIndex n, c, h, w;
    float alpha, decrement, apply_prob = 1., gamma;
    string data_format;
    vector<TIndex> seed_dims;
};

template <class Context>
class DropBlock2dGradientOp final : public Operator<Context> {
 public:
    DropBlock2dGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {
        SwitchToPhase(OperatorBase::Arg<string>("phase", ""));
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

DEFINE_ARGUMENT_WITH_DESC(float, DropBlock2dOp, keep_prob);

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_DROP_BLOCK_OP_H_