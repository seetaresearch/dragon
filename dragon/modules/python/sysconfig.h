/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_MODULES_PYTHON_SYSCONFIG_H_
#define DRAGON_MODULES_PYTHON_SYSCONFIG_H_

#include <dragon/core/context.h>
#include <dragon/utils/device/common_cuda.h>
#include <dragon/utils/device/common_cudnn.h>
#include <dragon/utils/device/common_eigen.h>
#include <dragon/utils/device/common_mlu.h>

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

void RegisterModule_sysconfig(py::module& m) {
  /*! \brief Set the logging severity */
  m.def("SetLoggingLevel", [](const string& severity) {
    SetLogDestination(severity);
  });

  /*! \brief Set the random seed for cuda device */
  m.def("SetRandomSeed", [](int seed) {
    CPUContext::objects().SetRandomSeed(seed);
  });

  /*! \brief Set the number of threads for cpu parallelism */
  m.def("SetNumThreads", [](int num) { Eigen::setNbThreads(num); });

  /*! \brief Return the number of threads for cpu parallelism */
  m.def("GetNumThreads", []() { return Eigen::nbThreads(); });

  m.def("GetBuildInformation", []() {
    static string build_info;
    if (!build_info.empty()) {
      return build_info;
    }
    // clang-format off
    build_info += "cpu_features:";
#if defined(USE_AVX)
    build_info += " AVX";
#endif
#if defined(USE_AVX2)
    build_info += " AVX2";
#endif
#if defined(USE_FMA)
    build_info += " FMA";
#endif
    build_info += "\ncuda_version:";
#if defined(USE_CUDA)
    build_info += " " + str::to(CUDA_VERSION / 1000) + "." +
                        str::to(CUDA_VERSION % 1000 / 10);
#endif
    build_info += "\ncudnn_version:";
#if defined(USE_CUDNN)
    build_info += " " + str::to(CUDNN_MAJOR) + "." +
                        str::to(CUDNN_MINOR) + "." +
                        str::to(CUDNN_PATCHLEVEL);
#endif
    build_info += "\nnccl_version:";
#if defined(USE_NCCL)
    build_info += " " + str::to(NCCL_MAJOR) + "." +
                        str::to(NCCL_MINOR) + "." +
                        str::to(NCCL_PATCH);
#endif
    build_info += "\nmps_version:";
#if defined(USE_MPS) && defined(MPS_OSX_VERSION_MAJOR)
    build_info += " " + str::to(MPS_OSX_VERSION_MAJOR) + "." +
                        str::to(MPS_OSX_VERSION_MINOR);
#endif
    build_info += "\ncnrt_version:";
#if defined(USE_MLU)
    build_info += " " + str::to(CNRT_MAJOR_VERSION) + "." +
                        str::to(CNRT_MINOR_VERSION) + "." +
                        str::to(CNRT_PATCH_VERSION);
#endif
    build_info += "\ncnnl_version:";
#if defined(USE_MLU)
    build_info += " " + str::to(CNNL_MAJOR) + "." +
                        str::to(CNNL_MINOR) + "." +
                        str::to(CNNL_PATCHLEVEL);
#endif
    build_info += "\ncncl_version:";
#if defined(USE_MLU)
    build_info += " " + str::to(CNCL_MAJOR_VERSION) + "." +
                        str::to(CNCL_MINOR_VERSION) + "." +
                        str::to(CNCL_PATCH_VERSION);
#endif
    build_info += "\nthird_party: eigen protobuf pybind11";
#if defined(USE_OPENMP)
    build_info += " openmp";
#endif
#if defined(USE_MPI)
    build_info += " mpi";
#endif
#if defined(USE_CUDA)
    build_info += " cuda cub";
#endif
#if defined(USE_CUDNN)
    build_info += " cudnn";
#endif
#if defined(USE_NCCL)
    build_info += " nccl";
#endif
#if defined(USE_MPS)
    build_info += " mps";
#endif
#if defined(USE_MLU)
    build_info += " cnrt cnnl cncl";
#endif
    // clang-format on
    return build_info;
  });
}

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_SYSCONFIG_H_
