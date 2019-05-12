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

#ifndef DRAGON_OPERATORS_NORM_L2_NORM_H_
#define DRAGON_OPERATORS_NORM_L2_NORM_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class L2NormOp final : public Operator<Context> {
 public:
    L2NormOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 0)),
          num_axes_(OpArg<int64_t>("num_axes", -1)),
          eps_(OpArg<float>("eps", 1e-5f)),
          mode_(OpArg<string>("mode", "SUM")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float eps_;
    string mode_;
    int64_t axis_, num_axes_, end_axis_;
    int64_t outer_dim_, reduce_dim_, inner_dim_;
};

template <class Context>
class L2NormGradientOp final : public Operator<Context> {
 public:
    L2NormGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 0)),
          num_axes_(OpArg<int64_t>("num_axes", -1)),
          mode_(OpArg<string>("mode", "SUM")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    string mode_;
    int64_t axis_, num_axes_, end_axis_;
    int64_t outer_dim_, reduce_dim_, inner_dim_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NORM_L2_NORM_H_