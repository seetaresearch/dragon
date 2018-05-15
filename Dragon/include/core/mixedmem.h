// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CORE_MIXEDMEM_H_
#define DRAGON_CORE_MIXEDMEM_H_

#include "context.h"
#include "context_cuda.h"

namespace dragon {

class MixedMemory {
 public:
    enum State { UNINITIALIZED, STATE_AT_CPU, STATE_AT_CUDA, SWITCHED, SYNCED };
    MixedMemory() : cpu_ptr_(nullptr), cuda_ptr_(nullptr) {}
    MixedMemory(const TypeMeta& meta, const size_t nbytes)
        : meta_(meta), nbytes_(nbytes),
        cpu_ptr_(nullptr), cuda_ptr_(nullptr) {}
    ~MixedMemory();

    const void* cpu_data();
    const void* cuda_data();
    void* mutable_cpu_data();
    void* mutable_cuda_data();
    void set_cpu_data(void* cpu_ptr, size_t nbytes);
#ifdef WITH_CUDA
    void async_cuda_data(const cudaStream_t& stream);
#endif

    void SwitchToDevice();
    void SwitchToCUDADevice(int device_id);

    inline size_t nbytes() const { return nbytes_; }

    inline void* cpu_ptr() { state_ = STATE_AT_CPU; return cpu_ptr_; }
    inline void* cuda_ptr() { state_ = STATE_AT_CUDA; return cuda_ptr_; }

    inline State state() const { return state_; }
    const Map<string, string> info() const;

    void ToCUDA();
    void ToCPU();

 private:
    void* cpu_ptr_, *cuda_ptr_;
    bool own_cpu_ptr_ = true;
    State state_ = UNINITIALIZED;
    size_t nbytes_ = 0;
    TypeMeta meta_;
};

}    // namespace dragon

#endif