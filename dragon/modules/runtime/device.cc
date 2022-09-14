#include <dragon/utils/logging.h>

#include "dragon/modules/runtime/dragon_runtime.h"

namespace dragon {

namespace {

inline std::string GetDeviceType(const std::string& device_type) {
  if (device_type == "CPU") {
    return "CPU";
  } else if (device_type == "GPU" || device_type == "CUDA") {
    return "CUDA";
  } else if (device_type == "MPS") {
    return "MPS";
  }
  LOG(FATAL) << "Unsupported device type: " << device_type << "\n"
             << "Following device types are supported: {"
             << "  * CPU\n"
             << "  * GPU\n"
             << "  * CUDA\n"
             << "  * MPS\n"
             << "}";
  return "";
}

} // namespace

Device::Device() : device_type_(0), device_index_(0) {}

Device::Device(const std::string& device_type, int device_index)
    : device_type_(GetDeviceType(device_type)), device_index_(device_index) {}

Device::Device(const std::string& device_type)
    : device_type_(GetDeviceType(device_type)), device_index_(0) {}

} // namespace dragon
