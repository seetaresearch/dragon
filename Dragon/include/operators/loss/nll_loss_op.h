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

#ifndef DRAGON_OPERATORS_LOSS_NLL_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_NLL_LOSS_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class NLLLossOp final : public Operator<Context> {
 public:
    NLLLossOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)),
          reduction_(OpArg<string>(
              "reduction", "VALID")) {
        auto ivec = OpArgs<int64_t>("ignore_labels");
        if (!ivec.empty()) {
            ignore_.Reshape({ (int64_t)ivec.size() });
            auto* idata = ignore_.mutable_data<int, CPUContext>();
            for (int i = 0; i < ivec.size(); i++) idata[i] = ivec[i];
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunImpl();

 protected:
    string reduction_;
    Tensor loss_, flag_, ignore_;
    int64_t axis_, outer_dim_, inner_dim_;
};

template <class Context>
class NLLLossGradientOp final : public Operator<Context> {
 public:
    NLLLossGradientOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)),
          reduction_(OpArg<string>(
              "reduction", "VALID")) {
        auto ivec = OpArgs<int64_t>("ignore_labels");
        if (!ivec.empty()) {
            ignore_.Reshape({ (int64_t)ivec.size() });
            auto* idata = ignore_.mutable_data<int, CPUContext>();
            for (int i = 0; i < ivec.size(); i++) idata[i] = ivec[i];
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunImpl();

 protected:
    string reduction_;
    Tensor ignore_, flag_;
    int64_t axis_, outer_dim_, inner_dim_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_LOSS_NLL_LOSS_OP_H_