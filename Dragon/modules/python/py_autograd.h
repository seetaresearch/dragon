/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_PYTHON_PY_AUTOGRAD_H_
#define DRAGON_PYTHON_PY_AUTOGRAD_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

void AddGradientMethods(pybind11::module& m) {
    m.def("CreateGradientDefs", [](
        const string&               forward_def,
        const vector<string>&       g_outputs) {
        OperatorDef def;
        if (!def.ParseFromString(forward_def))
            LOG(FATAL) << "Failed to parse the OperatorDef.";
        if (!GradientRegistry()->Has(def.type()))
            LOG(FATAL) << def.type() << "Op has no gradients.";
        Gradient grad = MakeGradientForOp(def, g_outputs);
        vector<pybind11::bytes> grad_ops;
        for (const auto& e : grad.ops)
            grad_ops.push_back(e.SerializeAsString());
        return std::tuple<
            vector<pybind11::bytes>, vector<string>, vector<float>
        >(grad_ops, grad.g_inputs, grad.defaults);
    });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_AUTOGRAD_H_