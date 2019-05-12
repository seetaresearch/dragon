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

#ifndef DRAGON_OPERATORS_ARRAY_ARGREDUCE_OP_H_
#define DRAGON_OPERATORS_ARRAY_ARGREDUCE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ArgReduceOp final : public Operator<Context> {
 public:
    ArgReduceOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          top_k_(OpArg<int64_t>("top_k", 1)),
          axis_(OpArg<int64_t>("axis", INT_MAX)),
          keep_dims_(OpArg<int64_t>("keep_dims", 0)),
          operation_(OpArg<string>("operation", "NONE")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    CPUContext cctx_;
    string operation_;
    int64_t axis_, top_k_, keep_dims_;
    int64_t outer_dim_, axis_dim_, inner_dim_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_ARGREDUCE_OP_H_