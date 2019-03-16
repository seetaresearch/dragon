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
        GET_ARGUMENTS_WITH_DESC(int64_t, starts);
        GET_ARGUMENTS_WITH_DESC(int64_t, sizes);
    }
    USE_OPERATOR_FUNCTIONS;

    void Setup();
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    vector<int64_t> st, ed, x_dimsV;
    Tensor startsT, y_stridesT, x_dimsT, fake_x;
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, starts);
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, sizes);
};

DEFINE_ARGUMENTS_WITH_DESC(int64_t, AssignOp, starts);
DEFINE_ARGUMENTS_WITH_DESC(int64_t, AssignOp, sizes);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_CONTROL_FLOW_ASSIGN_OP_H_