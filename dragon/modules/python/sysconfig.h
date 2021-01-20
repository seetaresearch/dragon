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

#include "dragon/modules/python/common.h"
#include "dragon/utils/device/common_eigen.h"

namespace dragon {

namespace python {

namespace sysconfig {

void RegisterModule(py::module& m) {
  /*! \brief Set the logging severity */
  m.def("SetLoggingLevel", [](const string& severity) {
    SetLogDestination(severity);
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
    build_info += " " + str::to(CUDNN_MAJOR) + "." + str::to(CUDNN_MINOR) +
        "." + str::to(CUDNN_PATCHLEVEL);
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
    return build_info;
  });
}

} // namespace sysconfig

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_SYSCONFIG_H_
