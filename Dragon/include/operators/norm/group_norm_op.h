// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_NORM_GROUP_NORM_OP_H_
#define DRAGON_OPERATORS_NORM_GROUP_NORM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class GroupNormOp : public Operator<Context> {
 public:
    GroupNormOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          group(OperatorBase::GetSingleArg<int>("group", 32)),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          eps(OperatorBase::GetSingleArg<float>("eps", 1e-3f)) {
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
    Tensor mean, num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier, *cgs_multiplier;
    Tensor* stddev, *var;
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    string data_format;
};

template <class Context>
class GroupNormGradientOp final : public Operator<Context> {
 public:
    GroupNormGradientOp(const OperatorDef& op_def, Workspace *ws)
        : Operator<Context>(op_def, ws),
          group(OperatorBase::GetSingleArg<int>("group", 32)),
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
    Tensor num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier, *cgs_multiplier;
    Tensor* stddev, *var;
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    string data_format;
};

template <class Context>
class FusedGroupNormOp : public Operator<Context> {
 public:
    FusedGroupNormOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          group(OperatorBase::GetSingleArg<int>("group", 32)),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          eps(OperatorBase::GetSingleArg<float>("eps", 1e-3f)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float eps;
    Tensor num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier, *cgs_multiplier;
    Tensor* mean, *var, *stddev, *x_norm;
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    string data_format;
};

template <class Context>
class FusedGroupNormGradientOp : public Operator<Context> {
 public:
    FusedGroupNormGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          group(OperatorBase::GetSingleArg<int>("group", 32)),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier, *cgs_multiplier;
    Tensor* mean, *var, *stddev, *x_norm;
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    string data_format;
};

}    // namespace dragon 

#endif    // DRAGON_OPERATORS_NORM_GROUP_NORM_OP_H_