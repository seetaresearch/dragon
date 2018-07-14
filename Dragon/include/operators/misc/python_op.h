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

#ifndef DRAGON_OPERATORS_MISC_PYTHON_OP_H_
#define DRAGON_OPERATORS_MISC_PYTHON_OP_H_

#ifdef WITH_PYTHON

#include <Python.h>

#include "core/operator.h"

namespace dragon {

template <class Context>
class RunOp : public Operator<Context> {
 public:
    RunOp(const OperatorDef& def, Workspace* ws);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    string CallMethodHelper(const string& method);
    PyObject* self, *inputs, *outputs;
    string module, op, param_str;
};

template <class Context>
class TemplateOp : public RunOp<Context> {
 public:
    TemplateOp(const OperatorDef& def, Workspace* ws)
        : RunOp<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class TemplateGradientOp final : public TemplateOp<Context> {
public:
    TemplateGradientOp(const OperatorDef& def, Workspace* ws)
        : TemplateOp<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
};

}    // namespace dragon

#endif    // WITH_PYTHON

#endif    // DRAGON_OPERATORS_MISC_PYTHON_OP_H_