#ifdef USE_PYTHON

#include "dragon/modules/python/plugin_op.h"

namespace dragon {

#ifdef USE_PYTHON3
#define PyBytes_FromStringAndSize PyUnicode_FromStringAndSize
#endif

#define PyBytes_FromRawString(raw_string) \
  PyBytes_FromStringAndSize(raw_string, string(raw_string).size())

#define PyBytes_FromStdString(std_string) \
  PyBytes_FromStringAndSize(std_string.c_str(), std_string.size())

template <class Context>
PythonPluginOp<Context>::PythonPluginOp(const OperatorDef& def, Workspace* ws)
    : Operator<Context>(def, ws),
      module_name_(OP_SINGLE_ARG(string, "module_name", "")),
      class_name_(OP_SINGLE_ARG(string, "class_name", "")),
      kwargs_str_(OP_SINGLE_ARG(string, "kwargs_str", "")) {
  // Initialize interpreter and load module.
  Py_Initialize();
  auto* module = PyImport_ImportModule(module_name_.c_str());
  CHECK(module) << "\nFailed to import module: " << module;
  auto* module_dict = PyModule_GetDict(module);
  auto* op_class = PyDict_GetItemString(module_dict, class_name_.c_str());
  CHECK(op_class) << "\nFailed to import class: " << class_name_
                  << " from module: " << module_name_;

  self_ = PyObject_CallObject(op_class, NULL);

  // Set inputs and outputs.
  inputs_ = PyList_New(InputSize());
  outputs_ = PyList_New(OutputSize());
  for (int i = 0; i < InputSize(); i++) {
    PyList_SetItem(inputs_, i, PyBytes_FromStdString(Input(i).name()));
  }
  for (int i = 0; i < OutputSize(); i++) {
    PyList_SetItem(outputs_, i, PyBytes_FromStdString(Output(i)->name()));
  }

  // Attr: "kwargs_str"
  PyObject_SetAttr(
      self_,
      PyBytes_FromRawString("kwargs_str"),
      PyBytes_FromStdString(kwargs_str_));

  if (PyObject_HasAttr(self_, PyBytes_FromRawString("setup"))) {
    CHECK(PyObject_CallMethod(self_, "setup", "OO", inputs_, outputs_))
        << CallMethodHelper("setup"); // Method: setup(inputs, outputs)
  }
}

template <class Context>
string PythonPluginOp<Context>::CallMethodHelper(const string& method_name) {
  std::stringstream ss;
  ss << "\nFailed to call: "
     << "<" + module_name_ << "." << class_name_ << "." << method_name
     << "(*args, **kwargs)>\n"
     << "This is a FATAL error to terminate "
     << "<" << name() << ">.";
  return ss.str();
}

template <class Context>
void PythonPluginOp<Context>::RunOnDevice() {
  // GIL may have been released.
  pybind11::gil_scoped_acquire g;

  // Attr: phase
  PyObject_SetAttr(
      self_, PyBytes_FromRawString("phase"), PyBytes_FromStdString(phase()));

  if (PyObject_HasAttr(self_, PyBytes_FromRawString("reshape"))) {
    CHECK(PyObject_CallMethod(self_, "reshape", "OO", inputs_, outputs_))
        << CallMethodHelper("reshape"); // Method: reshape(input, outputs)
  }

  if (PyObject_HasAttr(self_, PyBytes_FromRawString("forward"))) {
    CHECK(PyObject_CallMethod(self_, "forward", "OO", inputs_, outputs_))
        << CallMethodHelper("forward"); // Method: run(input, outputs)
  } else if (PyObject_HasAttr(self_, PyBytes_FromRawString("run"))) {
    CHECK(PyObject_CallMethod(self_, "run", "OO", inputs_, outputs_))
        << CallMethodHelper("run"); // Method: forward(input, outputs)
  }
}

DEPLOY_CPU_OPERATOR(PythonPlugin);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(PythonPlugin);
#endif

OPERATOR_SCHEMA(PythonPlugin);
NO_GRADIENT(PythonPlugin);

#undef PyBytes_FromRawString
#undef PyBytes_FromStdString

} // namespace dragon

#endif // USE_PYTHON
