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

#ifndef DRAGON_PYTHON_PY_CUDA_H_
#define DRAGON_PYTHON_PY_CUDA_H_

namespace dragon {

namespace python {

#include "py_dragon.h"

inline PyObject* IsCUDADriverSufficientCC(PyObject* self, PyObject* args) {
#ifdef WITH_CUDA
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorInsufficientDriver) return PyBool_FromLong(0);
    return PyBool_FromLong(1);
#else
    return PyBool_FromLong(0);
#endif
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_CUDA_H_