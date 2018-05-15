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

#ifndef DRAGON_OPERATORS_NORM_BATCH_RENORM_OP_H_
#define DRAGON_OPERATORS_NORM_BATCH_RENORM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class BatchRenormOp : public Operator<Context> {
 public:
    BatchRenormOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          momentum(OperatorBase::GetSingleArg<float>("momentum", float(0.9))),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          r_max(OperatorBase::GetSingleArg<float>("r_max", float(3.0))),
          d_max(OperatorBase::GetSingleArg<float>("d_max", float(5.0))),
          t_delta(OperatorBase::GetSingleArg<float>("t_delta", float(1.0))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)),
          t_r_max(float(1.0)), t_d_max(float(0.0)), t_val(float(0.0)),
          mode(OperatorBase::GetSingleArg<string>("mode", "DEFAULT")) {
        if (axis != -1)
            CHECK_EQ(axis, 1)
                << "\nThe axis can only be set to 1.";
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    float momentum, eps, r_max, d_max, t_delta;
    float t_r_max, t_d_max, t_val;
    Tensor mean, d, t_h_mean, t_h_var, num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier;
    Tensor* stddev, *r, *var, *x_norm;
    TIndex axis, N, C, S, NC, NS;
    string data_format, mode;
    int use_stats;
    bool use_global_stats, is_recomputing;
};

template <class Context>
class BatchRenormGradientOp final : public Operator<Context> {
 public:
    BatchRenormGradientOp(const OperatorDef& op_def, Workspace *ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) {
        if (axis != -1)
            CHECK_EQ(axis, 1)
                << "\nThe axis can only be set to 1.";
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

    template <typename T> void RunWithType();

 protected:
    Tensor mean, num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier;
    Tensor* stddev, *r, *var, *x_norm;
    TIndex axis, N, C, S, NC, NS;
    string data_format;
    int use_stats;
    bool use_global_stats;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_NORM_BATCH_RENORM_OP_H_