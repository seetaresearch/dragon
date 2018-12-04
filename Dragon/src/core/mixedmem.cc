#include "core/mixedmem.h"
#include "utils/cuda_device.h"
#include "utils/string.h"

namespace dragon {

void MixedMemory::ToCPU() {
    switch (state_) {
    case UNINITIALIZED:
        cpu_ptr_ = CPUContext::New(nbytes_);
        state_ = STATE_AT_CPU;
        break;
    case STATE_AT_CUDA:
#ifdef WITH_CUDA
        if (cpu_ptr_ == nullptr)
            cpu_ptr_ = CPUContext::New(nbytes_);
        CUDAContext::Memcpy<CPUContext, CUDAContext>(
            nbytes_, cpu_ptr_, cuda_ptr_);
        state_ = SYNCED;
#endif
        break;
    case STATE_AT_CPU:
    case SYNCED:
        break;
    }
}

void MixedMemory::ToCUDA() {
#ifdef WITH_CUDA
    void* new_ptr_ = nullptr;
    switch (state_) {
    case UNINITIALIZED:
        cuda_ptr_ = CUDAContext::New(nbytes_);
        ptr_device_ = CUDA_GET_DEVICE();
        state_ = STATE_AT_CUDA;
        break;
    case STATE_AT_CPU:
        if (cuda_ptr_ == nullptr) {
            cuda_ptr_ = CUDAContext::New(nbytes_);
            ptr_device_ = CUDA_GET_DEVICE();
        } CUDAContext::Memcpy<CUDAContext, CPUContext>(
            nbytes_, cuda_ptr_, cpu_ptr_);
        state_ = SYNCED;
        break;
    case SWITCHED:
        CHECK(cuda_ptr_) << "\nSwitched from an invalid cuda mem.";
        new_ptr_ = CUDAContext::New(nbytes_);
        CUDAContext::Memcpy<CUDAContext, CUDAContext>(
            nbytes_, new_ptr_, cuda_ptr_);
        CUDAContext::Delete(cuda_ptr_);
        cuda_ptr_ = new_ptr_;
        state_ = STATE_AT_CUDA;
        break;
    case STATE_AT_CUDA:
    case SYNCED:
        break;
    }
#endif
}

const void* MixedMemory::cpu_data() {
    ToCPU();
    return (const void*)cpu_ptr_;
}

const void* MixedMemory::cuda_data() {
    ToCUDA();
    return (const void*)cuda_ptr_;
}

const void* MixedMemory::cnml_data() {
    return (const void*)cnml_ptr_;
}

void* MixedMemory::mutable_cpu_data() {
    ToCPU();
    state_ = STATE_AT_CPU;
    return cpu_ptr_;
}

void* MixedMemory::mutable_cuda_data() {
    ToCUDA();
    state_ = STATE_AT_CUDA;
    return cuda_ptr_;
}

void* MixedMemory::mutable_cnml_data() {
    state_ = STATE_AT_CNML;
    return cnml_ptr_;
}

void MixedMemory::set_cpu_data(void* cpu_ptr, size_t nbytes) {
    bool use_cudahost_mem = false;
#ifdef WITH_CUDA_HOST_MEM
    use_cudahost_mem = true;
#endif
    if (own_cpu_ptr_ && cpu_ptr_ && !use_cudahost_mem) {
        if (meta_.dtor()) meta_.dtor()(
            cpu_ptr_, nbytes_ / meta_.itemsize());
        CPUContext::Delete(cpu_ptr_);
    }
#ifdef WITH_CUDA
    if (own_cpu_ptr_ && cpu_ptr_ &&
            use_cudahost_mem) cudaFreeHost(cpu_ptr_);
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
    bool use_cudahost_mem = false;
#ifdef WITH_CUDA_HOST_MEM
    use_cudahost_mem = true;
#endif
    if (own_cpu_ptr_ && cpu_ptr_ && !use_cudahost_mem) {
        if (meta_.dtor()) meta_.dtor()(
            cpu_ptr_, nbytes_ / meta_.itemsize());
        CPUContext::Delete(cpu_ptr_);
    }
#ifdef WITH_CUDA
    if (own_cpu_ptr_ && cpu_ptr_ &&
            use_cudahost_mem) cudaFreeHost(cpu_ptr_);
    if (cuda_ptr_) CUDAContext::Delete(cuda_ptr_);
#endif
}

void MixedMemory::SwitchToDevice() {
    if (cuda_ptr_) {
#ifdef WITH_CUDA
        int cur_device = CUDA_GET_DEVICE();
        if (cur_device != ptr_device_) {
            state_ = SWITCHED;
            ptr_device_ = cur_device;
        }
#endif
    }
}

void MixedMemory::SwitchToCUDADevice(int device_id) {
#ifdef WITH_CUDA
    DeviceGuard gurad(device_id);
    if (cuda_ptr_) {
        if (device_id != ptr_device_) {
            state_ = SWITCHED;
            ptr_device_ = device_id;
        }
    }
    ToCUDA();
#endif
}

const Map<string, string> MixedMemory::info() const {
    static map<State, string> STATE_TO_STRING {
        { UNINITIALIZED, "UNINITIALIZED" },
        { STATE_AT_CPU, "CPU" },
        { STATE_AT_CUDA, "CUDA" },
        { STATE_AT_CNML, "CNML" },
        { SYNCED, "DEVICE" },
        { SWITCHED, "DEVICE" },
    };
    Map<string, string> s2s;
    string _state_ = STATE_TO_STRING[state_];
    if (_state_ == "DEVICE") {
        if (cuda_ptr_) _state_ = "CUDA";
        else if (cnml_ptr_) _state_ = "CNML";
        else LOG(FATAL) << "Device activated, "
                        << "but got invalid mem pointer.";
    }
    s2s["mem_at"] = _state_;
    if (cpu_ptr_) s2s["CPU"] = "0";
    if (cuda_ptr_) s2s["CUDA"] = std::to_string(ptr_device_);
    else if (cnml_ptr_) s2s["CNML"] = std::to_string(ptr_device_);
    return s2s;
}

}  // namespace dragon