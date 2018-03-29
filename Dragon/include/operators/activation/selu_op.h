// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_OPERATORS_ACTIVATION_SELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_SELU_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class SEluOp : public Operator<Context> {
 public:
    SEluOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {}
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class SEluGradientOp : public Operator<Context> {
 public:
    SEluGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws) {
        DISABLE_SHARE_GRADIENT;
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ACTIVATION_SELU_OP_H_