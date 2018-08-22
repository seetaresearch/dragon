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

#ifndef DRAGON_PYTHON_PY_CONIFG_H_
#define DRAGON_PYTHON_PY_CONFIG_H_

#include "dragon.h"

inline PyObject* SetLogLevelCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "s", &cname)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted the logging level.");
        return nullptr;
    }
    SetLogDestination(StrToLogSeverity(string(cname)));
    Py_RETURN_TRUE;
}

#endif    // DRAGON_PYTHON_PY_CONFIG_H_