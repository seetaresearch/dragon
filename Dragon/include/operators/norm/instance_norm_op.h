// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_NORM_INSTANCE_NORM_OP_H_
#define DRAGON_OPERATORS_NORM_INSTANCE_NORM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class InstanceNormOp : public Operator<Context> {
 public:
    InstanceNormOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          inplace(OperatorBase::GetSingleArg<bool>("inplace", false)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float eps;
    Tensor mean;
    Tensor* spatial_multiplier, *stddev, *var;
    TIndex num, channels, spatial_dim, nbychans;
    bool inplace;
};

template <class Context>
class InstanceNormGradientOp final : public Operator<Context> {
 public:
    InstanceNormGradientOp(const OperatorDef& op_def, Workspace *ws) 
        : Operator<Context>(op_def, ws) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
     Tensor* spatial_multiplier, *stddev, *var;
     TIndex num, channels, spatial_dim, nbychans;
};
    
}    // namespace dragon

#endif    // DRAGON_OPERATORS_NORM_INSTANCE_NORM_OP_H_