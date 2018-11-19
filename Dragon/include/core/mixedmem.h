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

#include "core/context.h"
#include "core/context_cuda.h"
#include "core/context_cnml.h"

namespace dragon {

typedef enum {
    NCHW,
    NHWC,
} DataOrder;

class MixedMemory {
 public:
    typedef enum {
        UNINITIALIZED,
        STATE_AT_CPU,
        STATE_AT_CUDA,
        STATE_AT_CNML,
        SWITCHED,
        SYNCED,
    } State;

    MixedMemory() : cpu_ptr_(nullptr),
          cuda_ptr_(nullptr), cnml_ptr_(nullptr) {}
    MixedMemory(const TypeMeta& meta, const size_t nbytes)
        : meta_(meta), nbytes_(nbytes), cpu_ptr_(nullptr),
          cuda_ptr_(nullptr), cnml_ptr_(nullptr) {}
    ~MixedMemory();

    const void* cpu_data();
    const void* cuda_data();
    const void* cnml_data();

    void* mutable_cpu_data();
    void* mutable_cuda_data();
    void* mutable_cnml_data();

    void* malloc_cnml_data();
    void fetch_cnml_data(void** data);

    cnmlCpuTensor_t& cnml_cpu_tensor();
    cnmlTensor_t& cnml_mlu_tensor();

    void set_cpu_data(void* cpu_ptr, size_t nbytes);

    void SwitchToDevice();
    void SwitchToCUDADevice(int device_id);

    inline size_t nbytes() const { return nbytes_; }

    inline size_t nchunks() const { return nchunks_; }
    void set_nchunks(size_t nchunks) { nchunks_ = nchunks; }

    inline State state() const { return state_; }

    inline DataOrder order() const { return order_; }
    inline void set_order(DataOrder order) { order_ = order; }

    const Map<string, string> info() const;

    void ToCPU();
    void ToCUDA();

 private:
    TypeMeta meta_;

    size_t nbytes_ = 0, nchunks_ = 1;

    DataOrder order_ = NCHW;
    State state_ = UNINITIALIZED;

    void* cpu_ptr_, *cuda_ptr_, *cnml_ptr_;
    int own_cpu_ptr_ = 1, ptr_device_ = 0;

    /* For CAMBRICON's CNML Environment */
    cnmlCpuTensor_t cnml_cpu_tensor_ = nullptr;
    cnmlTensor_t cnml_mlu_tensor_ = nullptr;
};

}    // namespace dragon

#endif    // DRAGON_CORE_MIXEDMEM_H_