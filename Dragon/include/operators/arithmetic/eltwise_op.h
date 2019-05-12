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

#ifndef DRAGON_OPERATORS_ARITHMETIC_ELTWISE_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_ELTWISE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class EltwiseOp final : public Operator<Context> {
 public:
    EltwiseOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          coef_(OpArgs<float>("coef")),
          operation_(OpArg<string>("operation", "SUM")) {
        // Check the number of coeffients
        if (coef_.size() > 0) {
            CHECK_EQ(coef_.size(), XSize())
                << "\nOp has " << XSize() << " inputs, "
                << "while providing " << coef_.size() << " coefs.";
        } else {
            coef_.resize((size_t)XSize(), 1.f);
        }
        // Compute the alpha for product operation
        for (auto e : coef_) { if (e != 1.f) alpha_ *= e; }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
    template <typename T> void SumRunImpl();
    template <typename T> void ProdRunImpl();

 protected:
    string operation_;
    float alpha_ = 1.f;
    vector<float> coef_;
};

template <class Context>
class EltwiseGradientOp final : public Operator<Context> {
 public:
    EltwiseGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          coef_(OpArgs<float>("coef")),
          operation_(OpArg<string>("operation", "SUM")) {
        if (coef_.size() > 0) {
            CHECK_EQ(coef_.size(), YSize())
                << "\nOp has " << YSize() << " inputs, "
                << "while providing " << coef_.size() << " coefs.";
        } else {
            coef_.resize(YSize(), 1.f);
        }
        // Compute the alpha for product operation
        for (auto e : coef_) { if (e != 1.f) alpha_ *= e; }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
    template <typename T> void SumRunImpl();
    template <typename T> void ProdRunImpl();

 protected:
    string operation_;
    float alpha_ = 1.f;
    vector<float> coef_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_ELTWISE_OP_H_