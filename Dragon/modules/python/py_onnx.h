/*!
* Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
*
* Licensed under the BSD 2-Clause License.
* You should have received a copy of the BSD 2-Clause License
* along with the Xpensource.org/licenses/BSD-2-Clause>
*
* ------------------------------------------------------------
*/

#ifndef DRAGON_PYTHON_PY_ONNX_H_
#define DRAGON_PYTHON_PY_ONNX_H_

#include "contrib/onnx/onnx_backend.h"

#include "py_dragon.h"

namespace dragon {

namespace python {

inline PyObject* ImportONNXModelCC(PyObject* self, PyObject* args) {
    char* model_path;
    if (!PyArg_ParseTuple(args, "s", &model_path)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted the model path.");
        return nullptr;
    }
    GraphDef init_graph, pred_graph;
    onnx::ONNXBackend onnx_backend;
    onnx_backend.Prepare(model_path, &init_graph, &pred_graph);
    // Serializing to Python is intractable
    // We should apply the initializer immediately
    ws()->CreateGraph(init_graph);
    ws()->RunGraph(init_graph.name(), "", "");
    return String_AsPyBytes(pred_graph.SerializeAsString());
}

}  // namespace python

}  // namespace dragon

#endif // DRAGON_PYTHON_PY_ONNX_H_