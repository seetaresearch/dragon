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

#ifndef DRAGON_OPERATORS_VISION_DROP_BLOCK_OP_H_
#define DRAGON_OPERATORS_VISION_DROP_BLOCK_OP_H_

#include "core/operator.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
class DropBlock2dOp final
    : public Operator<Context> {
 public:
    DropBlock2dOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          alpha_(OpArg<float>("alpha", 1.f)),
          dec_(OpArg<float>("decrement", 0.f)),
          block_size_(OpArg<int64_t>("block_size", 7)) {
        GET_ARG_WITH_DESC(float, keep_prob, 0.9f);
        SwitchToPhase(OpArg<string>("phase", ""));
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    vec64_t seed_dims;
    int64_t n_, c_, h_, w_;
    int64_t block_size_, seed_h_, seed_w_;
    float alpha_, dec_, prob_ = 1., gamma_;
    DECLARE_ARG_WITH_DESC(float, keep_prob);
};

template <class Context>
class DropBlock2dGradientOp final
    : public Operator<Context> {
 public:
    DropBlock2dGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {
        SwitchToPhase(OpArg<string>("phase", ""));
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
};

DEFINE_ARG_WITH_DESC(float, DropBlock2dOp, keep_prob);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_DROP_BLOCK_OP_H_