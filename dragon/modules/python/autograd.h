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

#ifndef DRAGON_MODULES_PYTHON_AUTOGRAD_H_
#define DRAGON_MODULES_PYTHON_AUTOGRAD_H_

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

namespace autograd {

void RegisterModule(py::module& m) {
  m.def(
      "CreateGradientDef",
      [](const string& def_str, const vector<string>& grad_outputs) {
        OperatorDef def;
        CHECK(def.ParseFromString(def_str))
            << "\nFailed to parse the OperatorDef.";
        GradientPack pack = MakeGradientForOp(def, grad_outputs);
        vector<py::bytes> grad_defs;
        for (const auto& op_def : pack.grad_defs) {
          grad_defs.push_back(op_def.SerializeAsString());
        }
        return std::tuple<vector<py::bytes>, vector<string>, vector<float>>(
            grad_defs, pack.grad_inputs, pack.defaults);
      });
}

} // namespace autograd

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_AUTOGRAD_H_
