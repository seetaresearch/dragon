#include "core/mixedmem.h"
#include "utils/cuda_device.h"

namespace dragon {

void MixedMemory::ToCPU() {
    switch (state_) {
    case UNINITIALIZED:
        cpu_ptr_ = CPUContext::New(nbytes_);
        CPUContext::Memset(nbytes_, cpu_ptr_);
        state_ = STATE_AT_CPU;
        break;
    case STATE_AT_CUDA:
#ifdef WITH_CUDA
        if (cpu_ptr_ == nullptr)
            cpu_ptr_ = CPUContext::New(nbytes_);
        CUDAContext::Memcpy<CPUContext, CUDAContext>(nbytes_, cpu_ptr_, cuda_ptr_);
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
        CUDAContext::Memset(nbytes_, cuda_ptr_);
        state_ = STATE_AT_CUDA;
        break;
    case STATE_AT_CPU:
        if (cuda_ptr_ == nullptr)
            cuda_ptr_ = CUDAContext::New(nbytes_);
        CUDAContext::Memcpy<CUDAContext, CPUContext>(nbytes_, cuda_ptr_, cpu_ptr_);
        state_ = SYNCED;
        break;
    case SWITCHED:
        CHECK(cuda_ptr_) << "\nSwitched from an invalid cuda mem.";
        new_ptr_ = CUDAContext::New(nbytes_);
        CUDAContext::Memcpy<CUDAContext, CUDAContext>(nbytes_, new_ptr_, cuda_ptr_);
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

#ifdef WITH_CUDA
void MixedMemory::async_cuda_data(const cudaStream_t& stream) {
    CHECK(state_ == STATE_AT_CPU) << state_;
    if (cuda_ptr_ == NULL) cuda_ptr_ = CUDAContext::New(nbytes_);
    const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpyAsync(cuda_ptr_, cpu_ptr_, nbytes_, kind, stream));
    state_ = SYNCED;
}
#endif

MixedMemory::~MixedMemory() {
    bool use_cudahost_mem = false;
#ifdef WITH_CUDA_HOST_MEM
    use_cudahost_mem = true;
#endif
    if (cpu_ptr_ && !use_cudahost_mem) {
        if (meta_.dtor())
            meta_.dtor()(cpu_ptr_, nbytes_ / meta_.itemsize());
        CPUContext::Delete(cpu_ptr_);
    }
#ifdef WITH_CUDA
    if (cpu_ptr_ && use_cudahost_mem) cudaFreeHost(cpu_ptr_);
    if (cuda_ptr_) CUDAContext::Delete(cuda_ptr_);
#endif
}

void MixedMemory::SwitchToDevice() {
    if (cuda_ptr_) {
#ifdef WITH_CUDA
        int ptr_device = POINTER_DEVICE(cuda_ptr_);
        int cur_device = CURRENT_DEVICE();
        if (ptr_device != cur_device) state_ = SWITCHED;
#endif
    }
}

}    // namespace dragon