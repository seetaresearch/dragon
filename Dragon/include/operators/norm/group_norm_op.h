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
class GroupNormOp final : public Operator<Context> {
 public:
    GroupNormOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          group(OperatorBase::Arg<int>("group", 32)),
          axis(OperatorBase::Arg<int>("axis", -1)),
          eps(OperatorBase::Arg<float>("eps", 1e-3f)) {
        if (axis != -1) 
            CHECK_EQ(axis, 1) 
                << "\nThe axis can only be set to 1.";
    }
    USE_OPERATOR_FUNCTIONS;

    void Setup();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float eps;
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    Tensor nc, mean, *var;
    string data_format;
};

template <class Context>
class GroupNormGradientOp final : public Operator<Context> {
 public:
    GroupNormGradientOp(const OperatorDef& def, Workspace *ws)
        : Operator<Context>(def, ws),
          group(OperatorBase::Arg<int>("group", 32)),
          axis(OperatorBase::Arg<int>("axis", -1)) {
        if (axis != -1)
            CHECK_EQ(axis, 1)
                << "\nThe axis can only be set to 1.";
    }
    USE_OPERATOR_FUNCTIONS;

    void Setup();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    Tensor nc, *var;
    string data_format;
};

template <class Context>
class FusedGroupNormOp final : public Operator<Context> {
 public:
    FusedGroupNormOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          group(OperatorBase::Arg<int>("group", 32)),
          axis(OperatorBase::Arg<int>("axis", -1)),
          eps(OperatorBase::Arg<float>("eps", 1e-3f)) {}
    USE_OPERATOR_FUNCTIONS;

    void Setup();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    float eps;
    Tensor nc, *mean, *var, *x_norm;
    string data_format;
};

template <class Context>
class FusedGroupNormGradientOp final : public Operator<Context> {
 public:
    FusedGroupNormGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          group(OperatorBase::Arg<int>("group", 32)),
          axis(OperatorBase::Arg<int>("axis", -1)) {}
    USE_OPERATOR_FUNCTIONS;

    void Setup();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    Tensor nc, *mean, *var, *x_norm;
    string data_format;
};

}    // namespace dragon 

#endif    // DRAGON_OPERATORS_NORM_GROUP_NORM_OP_H_