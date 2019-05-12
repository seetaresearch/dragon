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

#ifndef DRAGON_OPERATORS_MISC_GRADIENT_OP_H_
#define DRAGON_OPERATORS_MISC_GRADIENT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class GradientGenerateOp final: public Operator<Context> {
 public:
    GradientGenerateOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          defaults(OpArgs<float>("defaults")) {
        CHECK_EQ(XSize(), YSize());
        CHECK_EQ(defaults.size(), YSize());
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    vector<float> defaults;
};

template <class Context>
class GradientGatherOp final : public Operator<Context> {
 public:
    GradientGatherOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {
        for (int i = 0; i < XSize(); i++) {
            if (X(i).name() != "NULL") {
                indices.push_back(i);
            }
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    vec32_t indices;
};

template <class Context>
class GradientAddOp final : public Operator<Context> {
 public:
    SIMPLE_CTOR_DTOR(GradientAddOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
};

template <class Context>
class StopGradientOp final : public Operator<Context> {
 public:
    SIMPLE_CTOR_DTOR(StopGradientOp);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_MISC_GRADIENT_OP_H_