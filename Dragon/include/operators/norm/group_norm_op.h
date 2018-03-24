// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

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
          momentum(OperatorBase::GetSingleArg<float>("momentum", float(0.9))),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)),
          mode(OperatorBase::GetSingleArg<string>("mode", "DEFAULT")) {
        if (axis != -1) 
            CHECK_EQ(axis, 1) 
                << "\nThe axis can only be set to 1.";
    }

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    float momentum, eps;
    Tensor mean, num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier, *cgs_multiplier;
    Tensor* stddev, *var;
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    string data_format, mode;
    int use_stats;
    bool use_global_stats, is_recomputing;
};

template <class Context>
class GroupNormGradientOp final : public Operator<Context> {
 public:
    GroupNormGradientOp(const OperatorDef& op_def, Workspace *ws)
        : Operator<Context>(op_def, ws),
          group(OperatorBase::GetSingleArg<int>("group", 32)),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) {
        if (axis != -1)
            CHECK_EQ(axis, 1)
                << "\nThe axis can only be set to 1.";
    }

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    Tensor num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier, *cgs_multiplier;
    Tensor* stddev, *var;
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    string data_format;
    int use_stats;
    bool use_global_stats;
};

template <class Context>
class FusedGroupNormOp : public Operator<Context> {
 public:
    FusedGroupNormOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          group(OperatorBase::GetSingleArg<int>("group", 32)),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          momentum(OperatorBase::GetSingleArg<float>("momentum", float(0.9))),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) {}

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    float momentum, eps;
    Tensor num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier, *cgs_multiplier;
    Tensor* mean, *var, *stddev, *x_norm;
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    string data_format;
    int use_stats;
    bool use_global_stats, is_recomputing;
};

template <class Context>
class FusedGroupNormGradientOp : public Operator<Context> {
 public:
    FusedGroupNormGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          group(OperatorBase::GetSingleArg<int>("group", 32)),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) {}

    void Setup();

    void ShareGradient() override;

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    float eps;
    Tensor num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier, *cgs_multiplier;
    Tensor* mean, *var, *stddev, *x_norm;
    TIndex group, axis, N, C, S, NG, NC, NS, CGS;
    string data_format;
    int use_stats;
    bool use_global_stats;
};

}    // namespace dragon 

#endif    // DRAGON_OPERATORS_NORM_GROUP_NORM_OP_H_