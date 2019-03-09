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

#ifndef DRAGON_PYTHON_PY_ONNX_H_
#define DRAGON_PYTHON_PY_ONNX_H_

#include "onnx/onnx_backend.h"

#include "py_dragon.h"

namespace dragon {

namespace python {

void AddONNXMethods(pybind11::module& m) {
    m.def("ImportONNXModel", [](
        const string&           model_path) {
        GraphDef init_graph, pred_graph;
        onnx::ONNXBackend onnx_backend;
        onnx_backend.Prepare(model_path, &init_graph, &pred_graph);
        // Serializing to Python is intractable
        // We should apply the initializer immediately
        ws()->CreateGraph(init_graph);
        ws()->RunGraph(init_graph.name(), "", "");
        return pybind11::bytes(pred_graph.SerializeAsString());
    });
}

}  // namespace python

}  // namespace dragon

#endif // DRAGON_PYTHON_PY_ONNX_H_