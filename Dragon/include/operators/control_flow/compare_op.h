// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_CONTROL_FLOW_COMPARE_OP_H_
#define DRAGON_OPERATORS_CONTROL_FLOW_COMPARE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class CompareOp final : public Operator<Context> {
 public:
    CompareOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          operation(OperatorBase::Arg<string>("operation", "NONE")) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void EqualRunWithType();

 protected:
    string operation;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_CONTROL_FLOW_COMPARE_OP_H_