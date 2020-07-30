#include "dragon/modules/runtime/dragon_runtime.h"
#include "dragon/utils/logging.h"

namespace dragon {

namespace {

int type_from_string(const std::string& device_type) {
  if (device_type == "CPU") {
    return 0;
  } else if (device_type == "GPU") {
    return 1;
  } else if (device_type == "CUDA") {
    return 1;
  }
  LOG(FATAL) << "Unsupported device type: " << device_type << "\n"
             << "Following device types are supported: {"
             << "  * CPU\n"
             << "  * GPU\n"
             << "  * CUDA\n"
             << "}";
  return -1;
}

} // namespace

Device::Device() : device_type_(0), device_id_(0) {}

Device::Device(const std::string& device_type, int device_id)
    : device_type_(type_from_string(device_type)), device_id_(device_id) {}

Device::Device(const std::string& device_type)
    : device_type_(type_from_string(device_type)), device_id_(0) {}

} // namespace dragon
