#include "dragon/core/memory.h"
#include "dragon/utils/device/common_cuda.h"

namespace dragon {

void UnifiedMemory::ToCPU(size_t size) {
  switch (state_) {
    case UNINITIALIZED:
      cpu_ptr_ = CPUContext::New(size_);
      CPUContext::Memset(size_, cpu_ptr_);
      state_ = STATE_AT_CPU;
      break;
    case STATE_AT_CUDA:
      if (cpu_ptr_ == nullptr) {
        cpu_ptr_ = CPUContext::New(size_);
      }
      CUDAContext::Memcpy<CPUContext, CUDAContext>(
          size > 0 ? size : size_, cpu_ptr_, cuda_ptr_, device_id_);
      state_ = SYNCED;
      break;
    case STATE_AT_CPU:
    case STATE_AT_CNML:
    case SYNCED:
      break;
  }
}

void UnifiedMemory::ToCUDA(size_t size) {
  switch (state_) {
    case UNINITIALIZED:
      cuda_ptr_ = CUDAContext::New(size_);
      CUDAContext::Memset(size_, cuda_ptr_);
      device_id_ = CUDAContext::current_device();
      state_ = STATE_AT_CUDA;
      break;
    case STATE_AT_CPU:
      if (cuda_ptr_ == nullptr) {
        cuda_ptr_ = CUDAContext::New(size_);
        device_id_ = CUDAContext::current_device();
      }
      CUDAContext::Memcpy<CUDAContext, CPUContext>(
          size > 0 ? size : size_, cuda_ptr_, cpu_ptr_, device_id_);
      state_ = SYNCED;
      break;
    case STATE_AT_CUDA:
    case STATE_AT_CNML:
    case SYNCED:
      break;
  }
}

const void* UnifiedMemory::cpu_data(size_t size) {
  ToCPU(size);
  return (const void*)cpu_ptr_;
}

const void* UnifiedMemory::cuda_data(size_t size) {
  SwitchToCUDADevice(CUDAContext::current_device());
  ToCUDA(size);
  return (const void*)cuda_ptr_;
}

const void* UnifiedMemory::cnml_data() {
  return (const void*)cnml_ptr_;
}

void* UnifiedMemory::mutable_cpu_data(size_t size) {
  ToCPU(size);
  state_ = STATE_AT_CPU;
  return cpu_ptr_;
}

void* UnifiedMemory::mutable_cuda_data(size_t size) {
  SwitchToCUDADevice(CUDAContext::current_device());
  ToCUDA(size);
  state_ = STATE_AT_CUDA;
  return cuda_ptr_;
}

void* UnifiedMemory::mutable_cnml_data() {
  state_ = STATE_AT_CNML;
  return cnml_ptr_;
}

void UnifiedMemory::set_cpu_data(void* cpu_ptr, size_t size) {
  if (own_cpu_ptr_ && cpu_ptr_) {
    if (meta_.dtor()) {
      meta_.dtor()(cpu_ptr_, size_ / meta_.itemsize());
    }
    CPUContext::Delete(cpu_ptr_);
  }
  size_ = size;
  cpu_ptr_ = cpu_ptr;
  state_ = STATE_AT_CPU;
  own_cpu_ptr_ = false;
}

void UnifiedMemory::set_cuda_data(void* cuda_ptr, size_t size, int device_id) {
  if (own_cuda_ptr_ && cuda_ptr_) {
    CUDAContext::Delete(cuda_ptr_);
  }
  size_ = size;
  cuda_ptr_ = cuda_ptr;
  state_ = STATE_AT_CUDA;
  own_cuda_ptr_ = false;
  device_id_ = device_id;
}

UnifiedMemory::~UnifiedMemory() {
  if (own_cpu_ptr_ && cpu_ptr_) {
    if (meta_.dtor()) {
      meta_.dtor()(cpu_ptr_, size_ / meta_.itemsize());
    }
    CPUContext::Delete(cpu_ptr_);
  }
  if (own_cuda_ptr_ && cuda_ptr_) {
    CUDAContext::Delete(cuda_ptr_);
  }
}

void UnifiedMemory::SwitchToCUDADevice(int device_id) {
#ifdef USE_CUDA
  if (cuda_ptr_) {
    if (device_id != device_id_) {
      void* new_ptr_ = nullptr;
      CUDADeviceGuard guard(device_id);
      new_ptr_ = CUDAContext::New(size_);
      CUDAContext::Memcpy<CUDAContext, CUDAContext>(
          size_, new_ptr_, cuda_ptr_, device_id_);
      if (own_cuda_ptr_) {
        CUDAContext::Delete(cuda_ptr_);
      }
      cuda_ptr_ = new_ptr_;
      device_id_ = device_id;
    }
  } else {
    CUDADeviceGuard guard(device_id);
    ToCUDA(size_);
  }
#endif
}

Map<string, string> UnifiedMemory::info() const {
  static map<State, string> state_map{
      {UNINITIALIZED, "uninitialized"},
      {STATE_AT_CPU, "cpu"},
      {STATE_AT_CUDA, "cuda"},
      {STATE_AT_CNML, "cnml"},
      {SYNCED, "device"},
  };
  string state_str = state_map[state_];
  if (state_str == "device") {
    if (cuda_ptr_) {
      state_str = "cuda";
    } else if (cnml_ptr_) {
      state_str = "cnml";
    } else {
      LOG(FATAL) << "Device activated, but got invalid mem pointer.";
    }
  }
  return {
      {"device_type", state_str},
      {"device_id", str::to(device_id_)},
  };
}

} // namespace dragon
