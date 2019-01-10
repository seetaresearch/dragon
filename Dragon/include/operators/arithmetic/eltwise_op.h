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
          operation(OperatorBase::Arg<string>("operation", "SUM")),
          coeffs(OperatorBase::Args<float>("coefficients")) {
        // Check the number of coeffients
        if (coeffs.size() > 0) {
            CHECK_EQ(coeffs.size(), InputSize())
                << "\nOp has " << InputSize() << " inputs, "
                << "but provided " << coeffs.size() << " coeffs.";
        } else coeffs.resize(InputSize(), 1.f);
        // Compute the alpha for product operation
        for (auto e : coeffs) { if (e != 1.f) alpha *= e; }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
    template <typename T> void SumRunWithType();
    template <typename T> void ProdRunWithType();

 protected:
    string operation;
    float alpha = 1.f;
    vector<float> coeffs;
};

template <class Context>
class EltwiseGradientOp final : public Operator<Context> {
 public:
    EltwiseGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          operation(OperatorBase::Arg<string>("operation", "SUM")),
          coeffs(OperatorBase::Args<float>("coefficients")) {
        if (coeffs.size() > 0) {
            CHECK_EQ(coeffs.size(), OutputSize())
                << "\nOp has " << OutputSize() << " inputs, "
                << "but provided " << coeffs.size() << " coeffs.";
        } else coeffs.resize(InputSize(), 1.f);
        // Compute the alpha for product operation
        for (auto e : coeffs) { if (e != 1.f) alpha *= e; }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();
    template <typename T> void SumRunWithType();
    template <typename T> void ProdRunWithType();

 protected:
    string operation;
    float alpha = 1.f;
    vector<float> coeffs;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_ELTWISE_OP_H_