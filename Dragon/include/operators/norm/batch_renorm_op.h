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

#ifndef DRAGON_OPERATORS_NORM_BATCH_RENORM_OP_H_
#define DRAGON_OPERATORS_NORM_BATCH_RENORM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class BatchRenormOp final : public Operator<Context> {
 public:
    BatchRenormOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", -1)),
          momentum(OperatorBase::Arg<float>("momentum", 0.9f)),
          eps(OperatorBase::Arg<float>("eps", 1e-5f)),
          r_max(OperatorBase::Arg<float>("r_max", 3.f)),
          d_max(OperatorBase::Arg<float>("d_max", 5.f)),
          t_delta(OperatorBase::Arg<float>("t_delta", 1.f)),
          use_stats(OperatorBase::Arg<int>("use_stats", -1)),
          t_r_max(1.f), t_d_max(0.f), t_val(0.f),
          mode(OperatorBase::Arg<string>("mode", "DEFAULT")) {
        if (axis != -1)
            CHECK_EQ(axis, 1)
                << "\nThe axis can only be set to 1.";
    }
    USE_OPERATOR_FUNCTIONS;

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    TIndex axis, use_stats, N, C, S, NC, NS;
    float momentum, eps, r_max, d_max, t_delta;
    float t_r_max, t_d_max, t_val;
    Tensor nc, mean, d, t_h_mean, t_h_var;
    Tensor* r, *var, *x_norm;
    string data_format, mode;
    bool use_global_stats, is_recomputing;
};

template <class Context>
class BatchRenormGradientOp final : public Operator<Context> {
 public:
    BatchRenormGradientOp(const OperatorDef& def, Workspace *ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", -1)),
          use_stats(OperatorBase::Arg<int>("use_stats", -1)) {
        if (axis != -1)
            CHECK_EQ(axis, 1)
                << "\nThe axis can only be set to 1.";
    }
    USE_OPERATOR_FUNCTIONS;

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

    template <typename T> void RunWithType();

 protected:
    TIndex axis, use_stats, N, C, S, NC, NS;
    Tensor nc, mean, *r, *var, *x_norm;
    string data_format;
    bool use_global_stats;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NORM_BATCH_RENORM_OP_H_