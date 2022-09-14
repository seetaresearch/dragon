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

#ifndef DRAGON_MODULES_PYTHON_MPS_H_
#define DRAGON_MODULES_PYTHON_MPS_H_

#include <dragon/core/context_mps.h>

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

void RegisterModule_mps(py::module& m) {
  /*! \brief Return whether mps driver is sufficient */
  m.def("mpsIsDriverSufficient", []() {
#ifdef USE_MPS
    int count = MTLGetDeviceCount();
    return count > 0;
#else
    return false;
#endif
  });

  /*! \brief Return the index of current device */
  m.def("mpsGetDevice", []() {
#ifdef USE_MPS
    return MPSContext::current_device();
#else
    return 0;
#endif
  });

  /*! \brief Return the capability of specified device */
  m.def("mpsGetDeviceName", [](int device_id) {
#ifdef USE_MPS
    if (device_id < 0) device_id = MPSContext::current_device();
    return MPSContext::objects().GetDeviceName(device_id);
#else
    return string("");
#endif
  });

  /*! \brief Return the capability of specified device */
  m.def("mpsGetDeviceFamily", [](int device_id) {
#ifdef USE_MPS
    if (device_id < 0) device_id = MPSContext::current_device();
    return MPSContext::objects().GetDeviceFamily(device_id);
#else
    return Set<string>();
#endif
  });

  /*! \brief Set the active mps device */
  m.def("mpsSetDevice", [](int device_id) {
#ifdef USE_MPS
    MPSContext::objects().SetDevice(device_id);
#endif
  });

  /*! \brief Synchronize the specified mps stream */
  m.def("mpsStreamSynchronize", [](int device_id, int stream_id) {
#ifdef USE_MPS
    if (device_id < 0) device_id = MPSContext::current_device();
    auto* stream = MPSContext::objects().stream(device_id, stream_id);
    MPSContext::SynchronizeStream(stream);
#endif
  });
}

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_MPS_H_
