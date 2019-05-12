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
          axis_(OpArg<int64_t>("axis", 1)),
          top_k_(OpArg<int64_t>("top_k", 1)) {
        auto ivec = OpArgs<int64_t>("ignore_labels");
        if (!ivec.empty()) {
            ignore_.Reshape({ (int64_t)ivec.size() });
            auto* x = ignore_.mutable_data<int, CPUContext>();
            for (int i = 0; i < ivec.size(); i++) x[i] = ivec[i];
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunImpl();

 protected:
    Tensor ignore_;
    CPUContext cctx_;
    int64_t outer_dim_, inner_dim_;
    int64_t axis_, axis_dim_, top_k_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_MISC_ACCURACY_OP_H_