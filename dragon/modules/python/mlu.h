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

#ifndef DRAGON_MODULES_PYTHON_MLU_H_
#define DRAGON_MODULES_PYTHON_MLU_H_

#include <dragon/core/context_mlu.h>

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

void RegisterModule_mlu(py::module& m) {
  /*! \brief Return whether MLU driver is sufficient */
  m.def("mluIsDriverSufficient", []() {
#ifdef USE_MLU
    int count = MLUGetDeviceCount();
    return count > 0;
#else
    return false;
#endif
  });

  /*! \brief Return the index of current device */
  m.def("mluGetDevice", []() {
#ifdef USE_MLU
    return MLUContext::current_device();
#else
    return 0;
#endif
  });

  /*! \brief Return the number of available devices */
  m.def("mluGetDeviceCount", []() {
#ifdef USE_MLU
    return MLUGetDeviceCount();
#else
    return 0;
#endif
  });

  /*! \brief Return the name of specified device */
  m.def("mluGetDeviceName", [](int device_id) {
#ifdef USE_MLU
    if (device_id < 0) device_id = MLUContext::current_device();
    auto& prop = MLUGetDeviceProp(device_id);
    return string(prop.name);
#else
    return string("");
#endif
  });

  /*! \brief Return the capability of specified device */
  m.def("mluGetDeviceCapability", [](int device_id) {
#ifdef USE_MLU
    if (device_id < 0) device_id = MLUContext::current_device();
    int major = 0, minor = 0;
    CNRT_CHECK(cnrtDeviceGetAttribute(
        &major, cnrtAttrComputeCapabilityMajor, device_id));
    CNRT_CHECK(cnrtDeviceGetAttribute(
        &minor, cnrtAttrComputeCapabilityMinor, device_id));
    return std::tuple<int, int>(major, minor);
#else
    return std::tuple<int, int>(0, 0);
#endif
  });

  /*! \brief Set the active mlu device */
  m.def("mluSetDevice", [](int device_id) {
#ifdef USE_MLU
    MLUContext::objects().SetDevice(device_id);
#endif
  });

  /*! \brief Synchronize the specified mlu stream */
  m.def("mluStreamSynchronize", [](int device_id, int stream_id) {
#ifdef USE_MLU
    if (device_id < 0) device_id = MLUContext::current_device();
    auto stream = MLUContext::objects().stream(device_id, stream_id);
    MLUContext::SynchronizeStream(stream);
#endif
  });

  /*! \brief Set the flags of CNNL library */
  m.def("cnnlSetFlags", [](int enabled) {
#ifdef USE_MLU
    auto& ctx = MLUContext::objects();
    if (enabled >= 0) ctx.cnnl_enabled_ = enabled;
#endif
  });
}

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_MLU_H_
