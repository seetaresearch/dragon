#include "operators/misc/python_op.h"

#ifdef WITH_PYTHON

#include <pybind11/pybind11.h>

#ifdef WITH_PYTHON3
#define PyBytes_FromStringAndSize \
    PyUnicode_FromStringAndSize
#endif

#define Bytes(str) \
    PyBytes_FromStringAndSize(str, string(str).size())

#define CS2Bytes(cstr) Bytes(cstr.c_str())

namespace dragon {

template <class Context>
RunOp<Context>::RunOp(const OperatorDef& def, Workspace* ws)
    : Operator<Context>(def, ws),
      module(OpArg<string>("module", "")),
      op(OpArg<string>("op", "")),
      param_str((OpArg<string>("param_str", ""))) {
    // Optimization for all python ops
    if (!AllowRun()) return; this->do_sync_ = false;

    // Init interpreter & load module
    Py_Initialize();
    PyObject* py_module = PyImport_ImportModule(module.c_str());
    CHECK(py_module) << "\nFailed to Import Module: " << module;
    PyObject* py_dict = PyModule_GetDict(py_module);
    PyObject* py_op = PyDict_GetItemString(py_dict, op.c_str());
    CHECK(py_op) << "\nFailed to Import Operator: " << op
                 << " from Module: " << module;
    self = PyObject_CallObject(py_op, NULL);

    // Wrap inputs and outputs
    inputs = PyList_New(XSize());
    for (int i = 0; i < XSize(); i++)
        PyList_SetItem(inputs, i,
            CS2Bytes(X(i).name()));

    outputs = PyList_New(YSize());
    for (int i = 0; i < YSize(); i++)
        PyList_SetItem(outputs, i,
            CS2Bytes(Y(i)->name()));

    // Backward compatibility: param_str
    PyObject_SetAttr(self, Bytes("param_str"), CS2Bytes(param_str));
    PyObject_SetAttr(self, Bytes("param_str_"), CS2Bytes(param_str));

    // Backward compatibility: self.setup(inputs, outputs)
    if (PyObject_HasAttr(self, Bytes("setup"))) {
        CHECK(PyObject_CallMethod(
            self, "setup", "OO",
                inputs, outputs
        )) << CallMethodHelper("setup");
    }
}

template <class Context>
string RunOp<Context>::CallMethodHelper(
    const string&           method) {
    std::stringstream ss;
    ss <<"\nFailed to call: "
       << "<" + module << "." << op
       << "." << method << "(*args, **kwargs)>\n"
       << "This is a FATAL error to terminate "
       << "<" << name() << ">.";
    return ss.str();
}

template <class Context>
void RunOp<Context>::RunOnDevice() {
    // GIL may have been released
    pybind11::gil_scoped_acquire g;

    // Reset phase
    PyObject_SetAttr(self, Bytes("phase"), CS2Bytes(phase()));

    // Backward compatibility: %self.reshape(*)%
    if (PyObject_HasAttr(self, Bytes("reshape"))) {
        CHECK(PyObject_CallMethod(
            self, "reshape", "OO",
                inputs, outputs
        )) << CallMethodHelper("reshape");
    }

    // Overloaded run inferfaces
    if (PyObject_HasAttr(self, Bytes("forward"))) {
        CHECK(PyObject_CallMethod(
            self, "forward", "OO",
                inputs, outputs
        )) << CallMethodHelper("forward");
    } else if (PyObject_HasAttr(self, Bytes("run"))) {
        CHECK(PyObject_CallMethod(
            self, "run", "OO",
                inputs, outputs
        )) << CallMethodHelper("run");
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
    // GIL may have been released
    pybind11::gil_scoped_acquire g;

    // Reset phase
    PyObject_SetAttr(this->self,
        Bytes("phase"), CS2Bytes(phase()));

    // Backward compatibility: %self.reshape(*)%
    if (PyObject_HasAttr(this->self, Bytes("reshape"))) {
        CHECK(PyObject_CallMethod(
            this->self, "reshape", "OO",
                this->inputs, this->outputs
        )) << this->CallMethodHelper("reshape");
    }

    // Overloaded run inferfaces
    if (PyObject_HasAttr(this->self, Bytes("backward"))) {
        CHECK(PyObject_CallMethod(
            this->self, "backward", "OO",
                this->inputs, this->outputs
        )) << this->CallMethodHelper("backward");
    } else if (PyObject_HasAttr(this->self, Bytes("grad"))) {
        CHECK(PyObject_CallMethod(
            this->self, "grad", "OO",
                this->inputs, this->outputs
        )) << this->CallMethodHelper("grad");
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

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        vector<string> inputs, outputs;
        for (auto input : def.input()) inputs.push_back(input);
        for (int i = 0; i < def.output_size(); i++) inputs.push_back(GO(i));
        for (int i = 0; i < def.input_size(); i++) outputs.push_back(GI(i));
        return SingleDef(def.type() + "Gradient", "", inputs, outputs);
    }
};

}  // namespace

REGISTER_GRADIENT(Template, GradientMaker);

}  // namespace dragon

#endif  // WITH_PYTHON