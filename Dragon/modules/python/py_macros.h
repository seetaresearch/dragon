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

#ifndef DRAGON_PYTHON_PY_MACROS_H_
#define DRAGON_PYTHON_PY_MACROS_H_

#include <string>
#include <sstream>

#include <Python.h>
#include <numpy/arrayobject.h>

namespace dragon {

namespace python {

#ifdef WITH_PYTHON3
#define PyInt_FromLong PyLong_FromLong
#define _PyInt_AsInt _PyLong_AsInt
#define PyString_AsString PyUnicode_AsUTF8
#endif

/*!
 * ------------------------------------------------------------
 *
 *                  <Having Fun with PyString>
 *
 * For Python3, Get/Return PyUnicode for regular string.
 * For Python3, Get/Return PyBytes for google-protobuf.
 * For Python2, Get/Return PyBytes only.
 *
 * ------------------------------------------------------------
 */

#define PyBytes_AsStringEx(pystring) \
    std::string(PyBytes_AsString(pystring), PyBytes_Size(pystring))

// Return string to Python
inline PyObject* String_AsPyBytes(const std::string& cstring) {
    return PyBytes_FromStringAndSize(cstring.c_str(), cstring.size());
}

inline PyObject* String_AsPyUnicode(const std::string& cstring) {
#ifdef WITH_PYTHON3
    return PyUnicode_FromStringAndSize(cstring.c_str(), cstring.size());
#else
    return PyBytes_FromStringAndSize(cstring.c_str(), cstring.size());
#endif
}

// Macors
#define PyList_AsVecString(plist, vs, defaults) \
    for (int i = 0; i < PyList_Size(plist); i++) { \
        PyObject* e = PyList_GetItem(plist, i); \
        if (e == Py_None) vs.emplace_back(defaults); \
        else vs.push_back(PyString_AsString(PyObject_Str(e))); \
    }

#define SetPyList(plist, ix, e) \
    PyList_SetItem(plist, ix, e)

#define SetPyDictS2S(object, key, value) \
    PyDict_SetItemString(object, key, Py_BuildValue("s", value))

#define SetPyDictS2I(object, key, value) \
    PyDict_SetItemString(object, key, Py_BuildValue("i", value))

// Misc
template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) { ss << t; }

template <typename T,typename ... Args>
inline void MakeStringInternal(std::stringstream& ss, const T& t, const Args& ... args) {
    MakeStringInternal(ss, t);
    MakeStringInternal(ss, args...);
}

template <typename ... Args>
std::string MakeString(const Args&... args) {
    std::stringstream ss;
    MakeStringInternal(ss, args...);
    return std::string(ss.str());
}

inline void PrErr_SetString(PyObject* type, const std::string& str) { 
    PyErr_SetString(type, str.c_str()); 
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_MACROS_H_