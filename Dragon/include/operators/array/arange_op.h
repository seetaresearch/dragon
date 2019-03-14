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

#ifndef DRAGON_OPERATORS_ARRAY_ARGMAX_OP_H_
#define DRAGON_OPERATORS_ARRAY_ARGMAX_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ArangeOp final : public Operator<Context> {
 public:
    ArangeOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          dtype(OperatorBase::Arg<string>("dtype", "float32")) {
        GET_ARGUMENT_WITH_DESC(int64_t, start, 0);
        GET_ARGUMENT_WITH_DESC(int64_t, stop, 0);
        GET_ARGUMENT_WITH_DESC(int64_t, step, 1);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    string dtype;
    int64_t astart, astop, astep, dim;
    DECLARE_ARGUMENT_WITH_DESC(int64_t, start);
    DECLARE_ARGUMENT_WITH_DESC(int64_t, stop);
    DECLARE_ARGUMENT_WITH_DESC(int64_t, step);
};

DEFINE_ARGUMENT_WITH_DESC(int64_t, ArangeOp, start);
DEFINE_ARGUMENT_WITH_DESC(int64_t, ArangeOp, stop);
DEFINE_ARGUMENT_WITH_DESC(int64_t, ArangeOp, step);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_ARANGE_OP_H_