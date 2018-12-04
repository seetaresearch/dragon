/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_CONTEXT_H_
#define DRAGON_CORE_CONTEXT_H_

#include "core/common.h"

namespace dragon {

class CPUContext {
 public:
    CPUContext(): random_seed_(3) {}
    CPUContext(unsigned int random_seed)
        : random_seed_(random_seed)  {}
    CPUContext(const DeviceOption& option)
        : random_seed_(option.has_random_seed() ?
            option.random_seed() : DEFAULT_RNG_SEED) {}
    virtual ~CPUContext() {}

    inline void SwitchToDevice() {}
    inline void SwitchToDevice(int stream_id) {}

    inline void FinishDeviceCompution() {}

    inline static void* New(size_t nbytes) {
        void* data;
#ifdef WITH_CUDA_HOST_MEM
        CUDA_CHECK(cudaMallocHost(&data, nbytes));
#else
        data = malloc(nbytes);
#endif
        CHECK(data) << "\nMalloc mem: " << nbytes << " bytes failed.";
        return data;
    }

    inline static void Memset(
        size_t              nbytes,
        void*               ptr) {
        memset(ptr, 0, nbytes);
    }

    inline void MemsetAsync(
        size_t              nbytes,
        void*               ptr) {
        memset(ptr, 0, nbytes);
    }

    template<class DstContext, class SrcContext>
    inline static void Memcpy(
        size_t              nbytes,
        void*               dst,
        const void*         src) {
        memcpy(dst, src, nbytes);
    }

    template<class DstContext, class SrcContext>
    inline void MemcpyAsync(
        size_t              nbytes,
        void*               dst,
        const void*         src) {
        memcpy(dst, src, nbytes);
    }

    template<typename T, class DstContext, class SrcContext>
    inline void Copy(
        int                 n,
        T*                  dst,
        const T*            src) {
        if (dst == src) return;
        //  only the basic types(e.g. int/float) can memcpy correctly
        if (std::is_fundamental<T>::value)
            Memcpy<DstContext, SrcContext>(
                n * sizeof(T), (void*)dst, (const void*)src);
        else for (int i = 0; i < n; i++) dst[i] = src[i];
    }

    inline static void Delete(void* data) { free(data); }

    inline int device_id() const { return 0; }
    inline void set_stream_id(int stream_id) {}

    inline std::mt19937* rand_generator() {
        if (!rand_generator_.get())
            rand_generator_.reset(new std::mt19937(random_seed_));
        return rand_generator_.get();
    }

 private:
    unsigned int random_seed_;
    unique_ptr<std::mt19937> rand_generator_;
};

#define CPU_FP16_NOT_SUPPORTED \
    LOG(FATAL) << "FP16 is unsupported for CPUContext.";

}  // namepsace dragon

#endif  // DRAGON_CORE_CONTEXT_H_