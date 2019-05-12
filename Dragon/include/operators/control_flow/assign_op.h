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

#ifndef DRAGON_OPERATORS_CONTROL_FLOW_ASSIGN_OP_H_
#define DRAGON_OPERATORS_CONTROL_FLOW_ASSIGN_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class AssignOp final : public Operator<Context> {
 public:
    AssignOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws) {
        GET_ARGS_WITH_DESC(int64_t, starts);
        GET_ARGS_WITH_DESC(int64_t, sizes);
    }
    USE_OPERATOR_FUNCTIONS;

    void Setup();
    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    vec64_t st_, ed_;
    Tensor X_, X_starts_, Y_strides_, X_dims_;
    DECLARE_ARGS_WITH_DESC(int64_t, starts);
    DECLARE_ARGS_WITH_DESC(int64_t, sizes);
};

DEFINE_ARGS_WITH_DESC(int64_t, AssignOp, starts);
DEFINE_ARGS_WITH_DESC(int64_t, AssignOp, sizes);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_CONTROL_FLOW_ASSIGN_OP_H_