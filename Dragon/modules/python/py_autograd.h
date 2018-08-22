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

#ifndef DRAGON_PYTHON_PY_AUTOGRAD_H_
#define DRAGON_PYTHON_PY_AUTOGRAD_H_

#include "dragon.h"
#include "core/graph_gradient.h"

PyObject* CreateGradientDefsCC(PyObject* self, PyObject* args) {
    PyObject* def_string = nullptr;
    PyObject* py_g_outputs = nullptr;
    if (!PyArg_ParseTuple(args, "SO!",
            &def_string, &PyList_Type, &py_g_outputs)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted a serialized string of OperatorDef "
            "and a list containing outputs of this GradientOp.");
         return nullptr;
    }
    OperatorDef def;
    if (!def.ParseFromString(PyBytes_AsStringEx(def_string))) {
        PyErr_SetString(PyExc_ValueError,
            "Failed to parse the OperatorDef.");
        return nullptr;
    }
    if (!GradientRegistry()->Has(def.type())) {
        PyErr_SetString(PyExc_KeyError,
            "This Operator does not register GradientOp.");
        return nullptr;
    }
    vector<string> g_outputs;
    PyList_AsVecString(py_g_outputs, g_outputs, "ignore");
    Gradient grad = MakeGradientForOp(def, g_outputs);
    PyObject* g_ops = PyList_New(grad.ops.size());
    PyObject* g_input = PyList_New(grad.g_inputs.size());
    PyObject* g_defaults = PyList_New(grad.defaults.size());
    for (int i = 0; i < grad.ops.size(); i++) {
        PyObject* e = String_AsPyBytes(grad.ops[i].SerializeAsString());
        SetPyList(g_ops, i, e);
    }
    for (int i = 0; i < grad.g_inputs.size(); i++) {
        PyObject* e = String_AsPyUnicode(grad.g_inputs[i]);
        SetPyList(g_input, i, e);
    }
    for (int i = 0; i < grad.defaults.size(); i++) {
        PyObject* e = PyFloat_FromDouble(grad.defaults[i]);
        SetPyList(g_defaults, i, e);
    }
    PyObject* pack = PyTuple_Pack(3, g_ops, g_input, g_defaults);
    Py_XDECREF(g_ops);
    Py_XDECREF(g_input);
    Py_XDECREF(g_defaults);
    return pack;
}

PyObject* RunGradientFlowCC(PyObject* self, PyObject* args) {
    PyObject* py_fp_ops, *py_targets;   
    PyObject* py_input_grads, *py_ignore_grads;
    PyObject* py_share_grads, *py_export_graph;
    if (!PyArg_ParseTuple(args, "OOOOOO",
        &py_fp_ops, &py_targets,
            &py_input_grads, &py_ignore_grads,
                &py_share_grads, &py_export_graph)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted a list of serialized input ops, targets, "
            "input grads, ignore grads and whehter to share grads or log graph.");
        return nullptr;
    }
    //  make & optm & run
    vector<string> targets, input_grads, ignore_grads;
    PyList_AsVecString(py_targets, targets, "");
    PyList_AsVecString(py_input_grads, input_grads, "");
    PyList_AsVecString(py_ignore_grads, ignore_grads, "");
    GraphDef fp_ops, bp_ops;
    if (!fp_ops.ParseFromString(PyBytes_AsStringEx(py_fp_ops))) {
        PyErr_SetString(PyExc_RuntimeError, 
            "Failed to parse the GraphDef of forward ops.");
        return nullptr;
    }
    GraphGradientMaker maker;
    for (auto& grad : input_grads) maker.AddExternalGrad(grad);
    for (auto& grad : ignore_grads) maker.AddIgnoreGrad(grad);
    maker.Make(fp_ops, targets, bp_ops);
    bool share_grads = PyObject_IsTrue(py_share_grads) ? true : false;
    bool export_graph = PyObject_IsTrue(py_export_graph) ? true : false;
    if (share_grads) maker.Share("/share/buffer/grads", bp_ops);
    if (export_graph) {
        Tensor* t = ws()->CreateTensor("/export/dynamic_graph/gradient_flow");
        t->Reshape({ 1 });
        string* data = t->mutable_data<string, CPUContext>();
        data[0] = bp_ops.SerializeAsString();
        t = ws()->CreateTensor("/export/dynamic_graph/forward_flow");
        t->Reshape({ 1 });
        data = t->mutable_data<string, CPUContext>();
        data[0] = fp_ops.SerializeAsString();
    }
    for (auto& op : bp_ops.op()) ws()->RunOperator(op);
    Py_RETURN_TRUE;
}

#endif    // DRAGON_PYTHON_PY_AUTOGRAD_H_