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
class NLLLossOp : public Operator<Context> {
 public:
    NLLLossOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 1)),
          normalization(OperatorBase::Arg<string>(
              "normalization", "VALID")) {
        auto xs = OperatorBase::Args<int64_t>("ignore_labels");
        if (xs.size()) {
            ignores.Reshape({ (int64_t)xs.size() });
            auto* Idata = ignores.mutable_data<int, CPUContext>();
            for (int i = 0; i < xs.size(); i++) Idata[i] = xs[i];
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunWithType();

 protected:
    int64_t axis, outer_dim, inner_dim;
    Tensor losses, flags, ignores;
    string normalization;
};

template <class Context>
class NLLLossGradientOp : public Operator<Context> {
 public:
    NLLLossGradientOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 1)),
          normalization(OperatorBase::Arg<string>(
              "normalization", "VALID")) {
        auto xs = OperatorBase::Args<int64_t>("ignore_labels");
        if (xs.size()) {
            ignores.Reshape({ (int64_t)xs.size() });
            auto* Idata = ignores.mutable_data<int, CPUContext>();
            for (int i = 0; i < xs.size(); i++) Idata[i] = xs[i];
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunWithType();

 protected:
    int64_t axis, outer_dim, inner_dim;
    Tensor ignores, flags;
    string normalization;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_LOSS_NLL_LOSS_OP_H_