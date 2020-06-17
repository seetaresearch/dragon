#ifdef USE_PYTHON

#include "dragon/modules/python/plugin_op.h"

#ifdef USE_PYTHON3
#define PyBytes_FromStringAndSize PyUnicode_FromStringAndSize
#endif

#define Bytes(str) PyBytes_FromStringAndSize(str, string(str).size())
#define S2Bytes(cstr) Bytes(cstr.c_str())

namespace dragon {

template <class Context>
PythonPluginInferOp<Context>::PythonPluginInferOp(
    const OperatorDef& def,
    Workspace* ws)
    : Operator<Context>(def, ws),
      module_name_(OpArg<string>("module_name", "")),
      class_name_(OpArg<string>("class_name", "")),
      kwargs_str_((OpArg<string>("kwargs_str", ""))) {
  // Optimization for all python ops
  if (!allow_run()) return;
  this->do_sync_ = false;

  // Initialize interpreter and load module
  Py_Initialize();
  auto* target_module = PyImport_ImportModule(module_name_.c_str());
  CHECK(target_module) << "\nFailed to import module: " << target_module;

  auto* module_dict = PyModule_GetDict(target_module);
  auto* target_class = PyDict_GetItemString(module_dict, class_name_.c_str());
  CHECK(target_class) << "\nFailed to import class: " << class_name_
                      << " from module: " << module_name_;

  self_ = PyObject_CallObject(target_class, NULL);

  // Project inputs and outputs
  inputs_ = PyList_New(InputSize());
  outputs_ = PyList_New(OutputSize());
  for (int i = 0; i < InputSize(); i++) {
    PyList_SetItem(inputs_, i, S2Bytes(Input(i).name()));
  }
  for (int i = 0; i < OutputSize(); i++) {
    PyList_SetItem(outputs_, i, S2Bytes(Output(i)->name()));
  }

  // Set: self.kwargs_str
  PyObject_SetAttr(self_, Bytes("kwargs_str"), S2Bytes(kwargs_str_));

  // Method: self.setup(inputs, outputs)
  if (PyObject_HasAttr(self_, Bytes("setup"))) {
    CHECK(PyObject_CallMethod(self_, "setup", "OO", inputs_, outputs_))
        << MethodHelper("setup");
  }
}

template <class Context>
string PythonPluginInferOp<Context>::MethodHelper(const string& method_name) {
  std::stringstream ss;
  ss << "\nFailed to call: "
     << "<" + module_name_ << "." << class_name_ << "." << method_name
     << "(*args, **kwargs)>\n"
     << "This is a FATAL error to terminate "
     << "<" << name() << ">.";
  return ss.str();
}

template <class Context>
void PythonPluginInferOp<Context>::RunOnDevice() {
  // GIL may have been released
  pybind11::gil_scoped_acquire g;

  // Reset: self.phase
  PyObject_SetAttr(self_, Bytes("phase"), S2Bytes(phase()));

  // Method: self.reshape(*)
  if (PyObject_HasAttr(self_, Bytes("reshape"))) {
    CHECK(PyObject_CallMethod(self_, "reshape", "OO", inputs_, outputs_))
        << MethodHelper("reshape");
  }

  // Method: self.run(input, outputs)
  // Method: self.forward(input, outputs)
  if (PyObject_HasAttr(self_, Bytes("forward"))) {
    CHECK(PyObject_CallMethod(self_, "forward", "OO", inputs_, outputs_))
        << MethodHelper("forward");
  } else if (PyObject_HasAttr(self_, Bytes("run"))) {
    CHECK(PyObject_CallMethod(self_, "run", "OO", inputs_, outputs_))
        << MethodHelper("run");
  }
}

template <class Context>
void PythonPluginGradientOp<Context>::RunOnDevice() {
  // GIL may have been released
  pybind11::gil_scoped_acquire g;

  // Reset: self.phase
  PyObject_SetAttr(this->self_, Bytes("phase"), S2Bytes(phase()));

  // Method: self.reshape(inputs, outputs)
  if (PyObject_HasAttr(this->self_, Bytes("reshape"))) {
    CHECK(PyObject_CallMethod(
        this->self_, "reshape", "OO", this->inputs_, this->outputs_))
        << this->MethodHelper("reshape");
  }

  // Method: self.grad(inputs, outputs)
  // Method: self.backward(inputs, outputs)
  if (PyObject_HasAttr(this->self_, Bytes("backward"))) {
    CHECK(PyObject_CallMethod(
        this->self_, "backward", "OO", this->inputs_, this->outputs_))
        << this->MethodHelper("backward");
  } else if (PyObject_HasAttr(this->self_, Bytes("grad"))) {
    CHECK(PyObject_CallMethod(
        this->self_, "grad", "OO", this->inputs_, this->outputs_))
        << this->MethodHelper("grad");
  }
}

DEPLOY_CPU(PythonPluginInfer);
#ifdef USE_CUDA
DEPLOY_CUDA(PythonPluginInfer);
#endif
OPERATOR_SCHEMA(PythonPluginInfer);

DEPLOY_CPU(PythonPlugin);
#ifdef USE_CUDA
DEPLOY_CUDA(PythonPlugin);
#endif
OPERATOR_SCHEMA(PythonPlugin);

DEPLOY_CPU(PythonPluginGradient);
#ifdef USE_CUDA
DEPLOY_CUDA(PythonPluginGradient);
#endif
OPERATOR_SCHEMA(PythonPluginGradient);

NO_GRADIENT(PythonPluginInfer);
REGISTER_GRADIENT(PythonPlugin, GenericGradientMaker);

} // namespace dragon

#endif // USE_PYTHON
