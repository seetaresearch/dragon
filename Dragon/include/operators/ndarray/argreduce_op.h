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

#ifndef DRAGON_OPERATORS_NDARRAY_ARGREDUCE_OP_H_
#define DRAGON_OPERATORS_NDARRAY_ARGREDUCE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ArgReduceOp final : public Operator<Context> {
 public:
    ArgReduceOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", -1)),
          operation(OperatorBase::Arg<string>("operation", "NONE")),
          keep_dims(OperatorBase::Arg<bool>("keep_dims", false)),
          top_k(OperatorBase::Arg<int>("top_k", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    bool keep_dims;
    string operation;
    TIndex axis, axis_dim, top_k, count, inner_dim;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_NDARRAY_ARGREDUCE_OP_H_