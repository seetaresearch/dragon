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

#ifndef DRAGON_OPERATORS_CONTROL_FLOW_COMPARE_OP_H_
#define DRAGON_OPERATORS_CONTROL_FLOW_COMPARE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CompareOp final : public Operator<Context> {
 public:
    CompareOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          op_str_(OpArg<string>("operation", "NONE")),
          to_uint8_(OpArg<bool>("to_uint8", false)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EqualRunImpl();
    template <typename T> void LessRunImpl();
    template <typename T> void LessEqualRunImpl();
    template <typename T> void GreaterRunImpl();
    template <typename T> void GreaterEqualRunImpl();
   
 protected:
    string op_str_;
    bool to_uint8_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_CONTROL_FLOW_COMPARE_OP_H_