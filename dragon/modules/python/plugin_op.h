/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_MODULES_PYTHON_PLUGIN_OP_H_
#define DRAGON_MODULES_PYTHON_PLUGIN_OP_H_

#ifdef USE_PYTHON

#include <pybind11/pybind11.h>
#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class PythonPluginInferOp : public Operator<Context> {
 public:
  PythonPluginInferOp(const OperatorDef& def, Workspace* ws);
  USE_OPERATOR_FUNCTIONS;

  string MethodHelper(const string& method);

  void RunOnDevice() override;

 protected:
  PyObject *self_, *inputs_, *outputs_;
  string module_name_, class_name_, kwargs_str_;
};

template <class Context>
class PythonPluginOp : public PythonPluginInferOp<Context> {
 public:
  PythonPluginOp(const OperatorDef& def, Workspace* ws)
      : PythonPluginInferOp<Context>(def, ws) {}
};

template <class Context>
class PythonPluginGradientOp final : public PythonPluginInferOp<Context> {
 public:
  PythonPluginGradientOp(const OperatorDef& def, Workspace* ws)
      : PythonPluginInferOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;
};

} // namespace dragon

#endif // USE_PYTHON

#endif // DRAGON_MODULES_PYTHON_PLUGIN_OP_H_
