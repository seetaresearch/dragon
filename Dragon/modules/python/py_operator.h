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

#ifndef DRAGON_PYTHON_PY_OPERATOR_H_
#define DRAGON_PYTHON_PY_OPERATOR_H_

#include "dragon.h"

inline PyObject* RegisteredOperatorsCC(PyObject* self, PyObject* args) {
    set<string> all_keys;
    for (const auto& name : CPUOperatorRegistry()->keys()) all_keys.insert(name);
    PyObject* list = PyList_New(all_keys.size());
    int idx = 0;
    for (const string& name : all_keys)
        CHECK_EQ(PyList_SetItem(list, idx++, String_AsPyUnicode(name)), 0);
    return list;
}

inline PyObject* NoGradientOperatorsCC(PyObject* self, PyObject* args) {
    set<string> all_keys;
    for (const auto& name : NoGradientRegistry()->keys()) all_keys.insert(name);
    PyObject* list = PyList_New(all_keys.size());
    int idx = 0;
    for (const string& name : all_keys)
        CHECK_EQ(PyList_SetItem(list, idx++, String_AsPyUnicode(name)), 0);
    return list;
}

inline PyObject* RunOperatorCC(PyObject* self, PyObject* args) {
    PyObject* op_str;
    if (!PyArg_ParseTuple(args, "S", &op_str)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted a serialized string of OperatorDef.");
        return nullptr;
    }
    OperatorDef op_def;
    if (!op_def.ParseFromString(PyBytes_AsStringEx(op_str))) {
        PyErr_SetString(PyExc_RuntimeError,
            "Failed to parse the OperatorDef.");
        return nullptr;
    }
    ws()->RunOperator(op_def);
    Py_RETURN_TRUE;
}

inline PyObject* RunOperatorsCC(PyObject* self, PyObject* args) {
    PyObject* py_ops;
    if (!PyArg_ParseTuple(args, "O", &py_ops)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted a list of serialized string of OperatorDef.");
        return nullptr;
    }
    OperatorDef op_def;
    for (int i = 0; i < PyList_Size(py_ops); i++) {
        PyObject* op_str = PyList_GetItem(py_ops, i);
        CHECK(op_def.ParseFromString(PyBytes_AsStringEx(op_str)));
        ws()->RunOperator(op_def);
    }
    Py_RETURN_TRUE;
}

inline PyObject* CreatePersistentOpCC(PyObject* self, PyObject* args) {
    PyObject* op_str;
    if (!PyArg_ParseTuple(args, "S", &op_str)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted a serialized string of OperatorDef.");
        return nullptr;
    }
    OperatorDef op_def;
    if (!op_def.ParseFromString(PyBytes_AsStringEx(op_str))) {
        PyErr_SetString(PyExc_RuntimeError,
            "Failed to parse the OperatorDef.");
        return nullptr;
    }
    ws()->CreatePersistentOp(op_def);
    Py_RETURN_TRUE;
}

inline PyObject* RunPersistentOpCC(PyObject* self, PyObject* args) {
    char* key, *anchor;
    PyObject* py_inputs, *py_outputs;
    if (!PyArg_ParseTuple(args, "ssOO",
            &key, &anchor, &py_inputs, &py_outputs)) {
        PyErr_SetString(PyExc_ValueError, 
            "Excepted a persistent key, anchor, "
            "list of inputs and outputs.");
        return nullptr;
    }
    vector<string> inputs, outputs;
    PyList_AsVecString(py_inputs, inputs, "");
    PyList_AsVecString(py_outputs, outputs, "");
    ws()->RunPersistentOp(key, anchor, inputs, outputs);
    Py_RETURN_TRUE;
}

#endif    // DRAGON_PYTHON_PY_OPERATOR_H_