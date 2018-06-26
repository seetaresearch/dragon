#include "operators/misc/python_op.h"

#ifdef WITH_PYTHON

#ifdef WITH_PYTHON3
#define PyBytes_FromStringAndSize PyUnicode_FromStringAndSize
#endif

#define String(str) \
    PyBytes_FromStringAndSize(str, string(str).size())

namespace dragon {

template <class Context>
RunOp<Context>::RunOp(const OperatorDef& op_def, Workspace* ws)
    : Operator<Context>(op_def, ws),
      module(OperatorBase::GetSingleArg<string>("module", "")),
      op(OperatorBase::GetSingleArg<string>("op", "")),
      param_str((OperatorBase::GetSingleArg<string>("param_str", ""))) {
    //  init interpreter & load module
    Py_Initialize();
    PyObject* py_module = PyImport_ImportModule(module.c_str());
    CHECK(py_module) << "\nFail to import py module: " << module;
    PyObject* py_dict = PyModule_GetDict(py_module);
    PyObject* py_op = PyDict_GetItemString(py_dict, op.c_str());
    CHECK(py_op) << "\nFail not import operator: " << op
                 << " from module: " << module;
    self = PyObject_CallObject(py_op, NULL);

    //  pass param string
    PyObject_SetAttr(self, String("param_str"), String(param_str.c_str()));
    PyObject_SetAttr(self, String("param_str_"), String(param_str.c_str()));

    //  build inputs and outputs for Python
    inputs = PyList_New(InputSize());
    for (int i = 0; i < InputSize(); i++)
        PyList_SetItem(inputs, i, String(Input(i).name().c_str()));
    outputs = PyList_New(OutputSize());
    for (int i = 0; i < OutputSize(); i++)
        PyList_SetItem(outputs, i, String(Output(i)->name().c_str()));
    if (!AllowRun()) return;

    //  setup
    if (PyObject_HasAttr(self, String("setup")))
        PyObject_CallMethod(self, "setup", "OO", inputs, outputs);
}

template <class Context>
void RunOp<Context>::RunOnDevice() {
    //  init phase
    PyObject_SetAttr(self, String("phase"), String(phase().c_str()));

    //  reshape
    if (PyObject_HasAttr(self, String("reshape")))
        PyObject_CallMethod(self, "reshape", "OO", inputs, outputs);

    //  run
    if (PyObject_HasAttr(self, String("forward"))) {
        PyObject_CallMethod(self, "forward", "OO", inputs, outputs);
    } else if (PyObject_HasAttr(self, String("run"))) {
        PyObject_CallMethod(self, "run", "OO", inputs, outputs);
    }
}

DEPLOY_CPU(Run);
#ifdef WITH_CUDA
DEPLOY_CUDA(Run);
#endif
OPERATOR_SCHEMA(Run);

NO_GRADIENT(Run);

template <class Context>
void TemplateGradientOp<Context>::RunOnDevice() {
    //  init phase
    PyObject_SetAttr(this->self, String("phase"), String(phase().c_str()));

    //  reshape
    if (PyObject_HasAttr(this->self, String("reshape")))
        PyObject_CallMethod(this->self, "reshape", "OO", this->inputs, this->outputs);

    //  run
    if (PyObject_HasAttr(this->self, String("backward"))) {
        PyObject_CallMethod(this->self, "forward", "OO", this->inputs, this->outputs);
    } else if (PyObject_HasAttr(this->self, String("grad"))) {
        PyObject_CallMethod(this->self, "grad", "OO", this->inputs, this->outputs);
    }
}

DEPLOY_CPU(Template);
#ifdef WITH_CUDA
DEPLOY_CUDA(Template);
#endif
OPERATOR_SCHEMA(Template);

DEPLOY_CPU(TemplateGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(TemplateGradient);
#endif
OPERATOR_SCHEMA(TemplateGradient);

class GetTemplateGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetTemplateGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs, outputs;
        for (auto input : def.input()) inputs.push_back(input);
        for (int i = 0; i < def.output_size(); i++) inputs.push_back(GO(i));
        for (int i = 0; i < def.input_size(); i++) outputs.push_back(GI(i));
        return SingleDef(def.type() + "Gradient", "", inputs, outputs);
    }
};
REGISTER_GRADIENT(Template, GetTemplateGradient);

}    // namespace dragon

#endif    // WITH_PYTHON