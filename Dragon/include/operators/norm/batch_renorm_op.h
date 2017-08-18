// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_NORM_BATCH_RENORM_OP_H_
#define DRAGON_OPERATORS_NORM_BATCH_RENORM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class BatchRenormOp : public Operator<Context> {
 public:
    BatchRenormOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          momentum(OperatorBase::GetSingleArg<float>("momentum", float(0.9))),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          r_max(OperatorBase::GetSingleArg<float>("r_max", float(3.0))),
          d_max(OperatorBase::GetSingleArg<float>("d_max", float(5.0))),
          t_delta(OperatorBase::GetSingleArg<float>("t_delta", float(1.0))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)),
          inplace(OperatorBase::GetSingleArg<bool>("inplace", false)),
          t_r_max(float(1.0)), t_d_max(float(0.0)), t_val(float(0.0)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float momentum, eps, r_max, d_max, t_delta;
    float t_r_max, t_d_max, t_val;
    Tensor mean, d, t_h_mean, t_h_var, num_by_chans;
    Tensor* num_multiplier, *spatial_multiplier;
    Tensor* stddev, *r, *var, *x_norm;
    TIndex num, channels, spatial_dim, nbychans;
    int use_stats;
    bool use_global_stats, inplace, is_recomputing;
};

template <class Context>
class BatchRenormGradientOp final : public Operator<Context> {
 public:
    BatchRenormGradientOp(const OperatorDef& op_def, Workspace *ws) 
        : Operator<Context>(op_def, ws),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor mean, num_by_chans;
    Tensor* num_multiplier, *spatial_multiplier;
    Tensor* stddev, *r, *var, *x_norm;
    TIndex num, channels, spatial_dim, nbychans;
    int use_stats;
    bool use_global_stats;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NORM_BATCH_RENORM_OP_H_