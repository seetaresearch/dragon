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

#ifndef DRAGON_OPERATORS_NORM_INSTANCE_NORM_OP_H_
#define DRAGON_OPERATORS_NORM_INSTANCE_NORM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class InstanceNormOp : public Operator<Context> {
 public:
    InstanceNormOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))) {
        if (axis != -1)
            CHECK_EQ(axis, 1)
                << "\nThe axis can only be set to 1.";
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float eps;
    Tensor mean;
    Tensor* spatial_multiplier, *stddev, *var;
    TIndex axis, N, C, S, NC, CS;
    string data_format;
};

template <class Context>
class InstanceNormGradientOp final : public Operator<Context> {
 public:
    InstanceNormGradientOp(const OperatorDef& op_def, Workspace *ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)) {
        if (axis != -1)
            CHECK_EQ(axis, 1)
                << "\nThe axis can only be set to 1.";
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();
                          
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
     Tensor* spatial_multiplier, *stddev, *var;
     TIndex axis, N, C, S, NC, CS;
     string data_format;
};
    
}    // namespace dragon

#endif    // DRAGON_OPERATORS_NORM_INSTANCE_NORM_OP_H_