#include "dragon.h"
#include "utils/logging.h"

namespace dragon {

int type_from_string(std::string type) {
    if (type == "CPU") return 0;
    else if (type == "GPU") return 1;
    else if (type == "CUDA") return 1;
    LOG(FATAL) << "Unknown device type: " << type << ", "
               << "known device types: "
               << "CPU, "
               << "GPU, "
               << "CUDA";
    return -1;
}

Device::Device()
    : device_type_(0), device_id_(0) {}

Device::Device(std::string device_type, int device_id)
    : device_type_(type_from_string(device_type)), device_id_(device_id) {}

Device::Device(std::string device_type)
    : device_type_(type_from_string(device_type)), device_id_(0) {}

}  // namespace dragon