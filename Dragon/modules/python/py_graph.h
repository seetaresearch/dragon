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

#ifndef DRAGON_PYTHON_PY_GRAPH_H_
#define DRAGON_PYTHON_PY_GRAPH_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

inline PyObject* CreateGraphCC(PyObject* self, PyObject* args) {
    PyObject* graph_str, *verbose;
    if (!PyArg_ParseTuple(args, "S|O", &graph_str, &verbose)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted a serialized string of GraphDef.");
        return nullptr;
    } 
    if (verbose == nullptr) verbose = Py_False;

    GraphDef graph_def;
    if (!graph_def.ParseFromString(PyBytes_AsStringEx(graph_str))) {
        PyErr_SetString(PyExc_RuntimeError,
            "Failed to parse the GraphDef.");
        return nullptr;
    } 

    auto* graph = ws()->CreateGraph(graph_def);

    if (!graph) {
        PyErr_SetString(PyExc_RuntimeError,
            "Failed to create the Graph.");
        return nullptr;
    } else {
        // It is not a good design to print the debug string
        if (PyObject_IsTrue(verbose) ? true : false) {
            auto* graph_tensor = ws()->CreateTensor(
                "/graph_def/optimized/" + graph->name());
            if (graph_tensor->count() > 0) {
                auto* data = graph_tensor->mutable_data<string, CPUContext>();
                std::cout << data[0] << std::endl;
            }
        }
    }
    // Return the graph name may be different from the def
    // We will make a unique dummy name on creating the graph
    return String_AsPyUnicode(graph->name());
}

inline PyObject* RunGraphCC(PyObject* self, PyObject* args) {
    char* cname, *include, *exclude;
    if (!PyArg_ParseTuple(args, "sss",
            &cname, &include, &exclude)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted the graph name, include and exclude rules.");
        return nullptr;
    }
    ws()->RunGraph(
        string(cname),
        string(include),
        string(exclude)
    );
    Py_RETURN_TRUE;
}

inline PyObject* GraphsCC(PyObject* self, PyObject* args) {
    vector<string> graphs = ws()->GetGraphs();
    PyObject* list = PyList_New(graphs.size());
    for (int i = 0; i < graphs.size(); i++)
        CHECK_EQ(PyList_SetItem(list, i, String_AsPyUnicode(graphs[i])), 0);
    return list;
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_GRAPH_H_