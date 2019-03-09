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

    m.def("FlowGradients", [](
        const vector<OperatorDef*>&   forward_ops,
        const vector<string>&         targets,
        const vector<string>&         input_grads,
        const vector<string>&         ignore_grads,
        const bool                    is_sharing,
        const bool                    verbose) {
        // Make => Optimize => Run
        GraphDef backward_ops;
        GraphGradientMaker maker;
        for (auto& grad : input_grads) maker.AddExternalGrad(grad);
        for (auto& grad : ignore_grads) maker.AddIgnoreGrad(grad);
        maker.Make(forward_ops, targets, backward_ops);
        if (is_sharing) maker.Share(backward_ops);
        pybind11::gil_scoped_release g;
        for (auto& op : backward_ops.op()) {
            if (verbose) std::cout << op.DebugString() << std::endl;
            if (op.has_uid()) ws()->RunOperator(op);
            else ws()->RunOperatorOnce(op);
        }
    });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_AUTOGRAD_H_