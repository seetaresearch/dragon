#include "core/mixedmem.h"
#include "utils/cuda_device.h"
#include "utils/string.h"

namespace dragon {

void MixedMemory::ToCPU(size_t nbytes) {
    switch (state_) {
        case UNINITIALIZED:
            cpu_ptr_ = CPUContext::New(nbytes_);
            CPUContext::Memset(nbytes_, cpu_ptr_);
            state_ = STATE_AT_CPU;
            break;
        case STATE_AT_CUDA:
            if (cpu_ptr_ == nullptr) {
                cpu_ptr_ = CPUContext::New(nbytes_);
            }
            CUDAContext::MemcpyEx<CPUContext, CUDAContext>(
                nbytes > 0 ? nbytes : nbytes_,
                cpu_ptr_, cuda_ptr_, ptr_device_);
            state_ = SYNCED;
            break;
        case STATE_AT_CPU:
        case SYNCED:
            break;
    }
}

void MixedMemory::ToCUDA(size_t nbytes) {
    switch (state_) {
        case UNINITIALIZED:
            cuda_ptr_ = CUDAContext::New(nbytes_);
            CUDAContext::Memset(nbytes_, cuda_ptr_);
            ptr_device_ = CUDAContext::active_device_id();
            state_ = STATE_AT_CUDA;
            break;
        case STATE_AT_CPU:
            if (cuda_ptr_ == nullptr) {
                cuda_ptr_ = CUDAContext::New(nbytes_);
                ptr_device_ = CUDAContext::active_device_id();
            }
            CUDAContext::MemcpyEx<CUDAContext, CPUContext>(
                nbytes > 0 ? nbytes : nbytes_,
                cuda_ptr_, cpu_ptr_, ptr_device_);
            state_ = SYNCED;
            break;
        case STATE_AT_CUDA:
        case SYNCED:
            break;
    }
}

const void* MixedMemory::cpu_data(size_t nbytes) {
    ToCPU(nbytes);
    return (const void*)cpu_ptr_;
}

const void* MixedMemory::cuda_data(size_t nbytes) {
    ToCUDA(nbytes);
    return (const void*)cuda_ptr_;
}

const void* MixedMemory::cnml_data() {
    return (const void*)cnml_ptr_;
}

void* MixedMemory::mutable_cpu_data(size_t nbytes) {
    ToCPU(nbytes);
    state_ = STATE_AT_CPU;
    return cpu_ptr_;
}

void* MixedMemory::mutable_cuda_data(size_t nbytes) {
    ToCUDA(nbytes);
    state_ = STATE_AT_CUDA;
    return cuda_ptr_;
}

void* MixedMemory::mutable_cnml_data() {
    state_ = STATE_AT_CNML;
    return cnml_ptr_;
}

void MixedMemory::set_cpu_data(void* cpu_ptr, size_t nbytes) {
    if (own_cpu_ptr_ && cpu_ptr_) {
        if (meta_.dtor()) meta_.dtor()(
            cpu_ptr_, nbytes_ / meta_.itemsize());
        CPUContext::Delete(cpu_ptr_);
    }
#ifdef WITH_CUDA
    if (cuda_ptr_ && nbytes > nbytes_) {
        // Maintain the cuda ptr as regular mems
        CUDAContext::Delete(cuda_ptr_);
        cuda_ptr_ = nullptr;
    }
#endif
    cpu_ptr_ = cpu_ptr;
    nbytes_ = nbytes;
    state_ = STATE_AT_CPU;
    own_cpu_ptr_ = false;
}

MixedMemory::~MixedMemory() {
    if (own_cpu_ptr_ && cpu_ptr_) {
        if (meta_.dtor()) meta_.dtor()(
            cpu_ptr_, nbytes_ / meta_.itemsize());
        CPUContext::Delete(cpu_ptr_);
    }
#ifdef WITH_CUDA
    if (cuda_ptr_) CUDAContext::Delete(cuda_ptr_);
#endif
}

void MixedMemory::SwitchToDevice(int device_id) {
    if (cuda_ptr_) {
        SwitchToCUDADevice(device_id);
    }
}

void MixedMemory::SwitchToCUDADevice(int device_id) {
#ifdef WITH_CUDA
    if (cuda_ptr_) {
        if (device_id != ptr_device_) {
            // Move the memory to another device
            void* new_ptr_ = nullptr;
            DeviceGuard gurad(device_id);
            new_ptr_ = CUDAContext::New(nbytes_);
            CUDAContext::MemcpyEx<CUDAContext, CUDAContext>(
                nbytes_, new_ptr_, cuda_ptr_, ptr_device_);
            CUDAContext::Delete(cuda_ptr_);
            // Update the pointer
            cuda_ptr_ = new_ptr_;
            ptr_device_ = device_id;
        }
    }
#endif
}

const Map<string, string> MixedMemory::info() const {
    static map<State, string> STATE_TO_STRING {
        { UNINITIALIZED, "uninitialized" },
        { STATE_AT_CPU, "cpu" },
        { STATE_AT_CUDA, "cuda" },
        { STATE_AT_CNML, "cnml" },
        { SYNCED, "device" },
    };
    Map<string, string> s2s;
    string _state_ = STATE_TO_STRING[state_];
    if (_state_ == "device") {
        if (cuda_ptr_) _state_ = "cuda";
        else if (cnml_ptr_) _state_ = "cnml";
        else LOG(FATAL) << "Device activated, "
                        << "but got invalid mem pointer.";
    }
    s2s["device_type"] = _state_;
    s2s["device_id"] = std::to_string(ptr_device_);
    return s2s;
}

}  // namespace dragon