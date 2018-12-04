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

#ifndef DRAGON_PYTHON_PY_IO_H_
#define DRAGON_PYTHON_PY_IO_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

inline PyObject* SnapshotCC(PyObject* self, PyObject* args) {
    char* path; int format;
    PyObject* names; vector<Tensor*> tensors;
    if (!PyArg_ParseTuple(args, "sOi", &path, &names, &format)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted the model path, tensors, and data format.");
        return nullptr;
    }
    switch (format) {
        case 0:  // Pickle
            PyErr_SetString(PyExc_NotImplementedError,
                "Format depends on Pickle. Can't be used in C++.");
            break;
        case 1:  // CaffeModel
            for (int i = 0; i < PyList_Size(names); i++)
                tensors.push_back(ws()->GetTensor(
                    PyString_AsString(PyList_GetItem(names, i))));
            SavaCaffeModel(path, tensors);
            break;
        default: LOG(FATAL) << "Unknwon format, code: " << format;
   }
   Py_RETURN_TRUE;
}

inline PyObject* RestoreCC(PyObject* self, PyObject* args) {
    char* path; int format;
    if (!PyArg_ParseTuple(args, "si", &path, &format)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted the model path and data format.");
        return nullptr;
    }
    switch (format) {
        case 0:  // Pickle
            PyErr_SetString(PyExc_NotImplementedError,
                "Format depends on Pickle. Can't be used in C++.");
            break;
        case 1:  // CaffeModel
            LoadCaffeModel(path, ws());
            break;
        default: LOG(FATAL) << "Unknwon format, code: " << format;
    }
    Py_RETURN_TRUE;
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_IO_H_