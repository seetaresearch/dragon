/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_MODULES_PYTHON_PLUGIN_OP_H_
#define DRAGON_MODULES_PYTHON_PLUGIN_OP_H_

#ifdef USE_PYTHON

#include <dragon/core/operator.h>
#include <pybind11/pybind11.h>

namespace dragon {

template <class Context>
class PythonPluginOp : public Operator<Context> {
 public:
  PythonPluginOp(const OperatorDef& def, Workspace* ws);
  USE_OPERATOR_FUNCTIONS;

  string CallMethodHelper(const string& method_name);

  void RunOnDevice() override;

 protected:
  PyObject *self_, *inputs_, *outputs_;
  string module_name_, class_name_, kwargs_str_;
};

} // namespace dragon

#endif // USE_PYTHON

#endif // DRAGON_MODULES_PYTHON_PLUGIN_OP_H_
