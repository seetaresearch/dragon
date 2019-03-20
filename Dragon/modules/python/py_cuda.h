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

void AddCUDAMethods(pybind11::module& m) {
    m.def("IsCUDADriverSufficient", []() {
#ifdef WITH_CUDA
        int count;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err == cudaErrorInsufficientDriver) false;
        return true;
#else
        return false;
#endif
    });

    m.def("EnableCUDNN", [](bool enabled) {
#ifdef WITH_CUDA
        CUDAContext::cuda_object()
            ->cudnn_enabled = enabled;
#endif
    });

    m.def("cudaGetDevice", []() {
        return CUDAContext::active_device_id();
    });

    m.def("cudaStreamSynchronize", [](
        int device_id, int stream_id) {
#ifdef WITH_CUDA
        if (device_id < 0) device_id =
            CUDAContext::active_device_id();
        cudaStreamSynchronize(CUDAContext::cuda_object()
            ->GetStream(device_id, stream_id));
        cudaError_t error = cudaGetLastError();
        CHECK_EQ(error, cudaSuccess)
            << "\nCUDA Error: " << cudaGetErrorString(error);
#endif
    });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_CUDA_H_