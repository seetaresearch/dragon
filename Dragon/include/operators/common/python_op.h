// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_COMMON_PYTHON_OP_H_
#define DRAGON_OPERATORS_COMMON_PYTHON_OP_H_

#include <Python.h>

#include "core/operator.h"

#ifdef WITH_PYTHON3
#define PyBytes_FromStringAndSize PyUnicode_FromStringAndSize
#endif

namespace dragon {

template <class Context>
class RunOp : public Operator<Context> {
 public:
    RunOp(const OperatorDef& op_def, Workspace* ws);
    PyObject* String(const char* str) {
        return PyBytes_FromStringAndSize(str, string(str).size());
    }

    void RunOnDevice() override;

 protected:
    PyObject* self, *inputs, *outputs;
    string module, op, param_str;
};

template <class Context>
class TemplateOp : public RunOp<Context> {
 public:
    TemplateOp(const OperatorDef& op_def, Workspace* ws)
        : RunOp<Context>(op_def, ws) {}
};

template <class Context>
class TemplateGradientOp : public TemplateOp<Context> {
public:
    TemplateGradientOp(const OperatorDef& op_def, Workspace* ws)
        : TemplateOp<Context>(op_def, ws) {}
    void RunOnDevice() override;
};


}    // namespace dragon

#endif    // DRAGON_OPERATORS_COMMON_PYTHON_OP_H_