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

#ifndef DRAGON_OPERATORS_MISC_ACCURACY_OP_H_
#define DRAGON_OPERATORS_MISC_ACCURACY_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class AccuracyOp final : public Operator<Context> {
 public:
    AccuracyOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          top_k(OperatorBase::Arg<int64_t>("top_k", 1)),
          axis(OperatorBase::Arg<int64_t>("axis", 1)) {
        auto ignores = OperatorBase::Args<int>("ignore_labels");
        if (ignores.size()) {
            ignore.Reshape({ (int64_t)ignores.size() });
            auto* Idata = ignore.mutable_data<int, CPUContext>();
            for (int i = 0; i < ignores.size(); i++) Idata[i] = ignores[i];
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunWithType();

 protected:
    int64_t top_k, axis, outer_dim, inner_dim, num_classes;
    Tensor ignore;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_MISC_ACCURACY_OP_H_