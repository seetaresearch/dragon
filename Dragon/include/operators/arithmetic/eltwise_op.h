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

#ifndef DRAGON_OPERATORS_ARITHMETIC_ELTWISE_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_ELTWISE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class EltwiseOp final : public Operator<Context> {
 public:
    EltwiseOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          operation(OperatorBase::GetSingleArg<string>("operation", "SUM")),
          coeffs(OperatorBase::GetRepeatedArg<float>("coeffs")) {
        if (coeffs.size() > 0) {
            CHECK_EQ(coeffs.size(), InputSize())
                << "\nOp has " << InputSize() << " inputs, "
                << "but provided " << coeffs.size() << " coeffs.";
        } else coeffs.resize(InputSize(), float(1));
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void SumRunWithType();
    template <typename T> void ProdRunWithType();

 protected:
    string operation;
    vector<float> coeffs;
};

template <class Context>
class EltwiseGradientOp final : public Operator<Context> {
 public:
    EltwiseGradientOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          operation(OperatorBase::GetSingleArg<string>("operation", "SUM")),
          coeffs(OperatorBase::GetRepeatedArg<float>("coeff")) {
        if (coeffs.size() > 0) {
            CHECK_EQ(coeffs.size(), InputSize())
                << "\nop has " << InputSize() << " inputs, "
                << "but provided " << coeffs.size() << " coeffs.";
        } else coeffs.resize(InputSize(), float(1));
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void SumRunWithType();
    template <typename T> void ProdRunWithType();

 protected:
    string operation;
    vector<float> coeffs;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_ELTWISE_OP_H_