// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_PYTHON_PY_GRAPH_H_
#define DRAGON_PYTHON_PY_GRAPH_H_

#include "dragon.h"

inline PyObject* CreateGraphCC(PyObject* self, PyObject* args) {
    PyObject* graph_str;
    if (!PyArg_ParseTuple(args, "S", &graph_str)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted a serialized string of GraphDef.");
        return nullptr;
    }
    GraphDef graph_def;
    if (!graph_def.ParseFromString(PyBytes_AsStringEx(graph_str))) {
        PyErr_SetString(PyExc_RuntimeError,
            "Failed to parse the GraphDef.");
        return nullptr;
    } 
    if (!ws()->CreateGraph(graph_def)) {
        PyErr_SetString(PyExc_RuntimeError,
            "Failed to create the Graph.");
        return nullptr;
    }
    Py_RETURN_TRUE;
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

#endif    // DRAGON_PYTHON_PY_GRAPH_H_