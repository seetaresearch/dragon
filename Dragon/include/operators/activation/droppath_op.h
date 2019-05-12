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

#ifndef DRAGON_OPERATORS_ACTIVATION_DROPPATH_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_DROPPATH_OP_H_

#include "core/operator.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
class DropPathOp final : public Operator<Context> {
 public:
    DropPathOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          inc_(OpArg<float>("increment", 0.f)) {
        SwitchToPhase(OpArg<string>("phase", ""));
        GET_ARG_WITH_DESC(float, prob, 0.2f);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t rows_, cols_;
    float inc_, drop_prob_ = 0.f;
    DECLARE_ARG_WITH_DESC(float, prob);
};

template <class Context>
class DropPathGradientOp final : public Operator<Context> {
 public:
    DropPathGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {
        SwitchToPhase(OpArg<string>("phase", ""));
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t rows_, cols_;
};

DEFINE_ARG_WITH_DESC(float, DropPathOp, prob);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ACTIVATION_DROPPATH_OP_H_