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
    /*! \brief Default Constructor */
    CPUContext(): random_seed_(3) {}

    /*! \brief Constructor with the specified random seed */
    CPUContext(unsigned int random_seed)
        : random_seed_(random_seed) {}

    /*! \brief Constructor with the specified device option */
    CPUContext(const DeviceOption& option)
        : random_seed_(option.has_random_seed() ?
            option.random_seed() : DEFAULT_RNG_SEED) {}

    /*! \brief Deconstructor */
    virtual ~CPUContext() {}

    /*! \brief Switch to the device of this context */
    void SwitchToDevice() {}

    /*! \brief Switch to the device with the given stream */
    void SwitchToDevice(const int stream_id) {}

    /*! \brief Synchronize the dispatched operations */
    void FinishDeviceCompution() {}

    /*! \brief Malloc the memory */
    static void* New(size_t nbytes) {
        void* data;
#ifdef WITH_CUDA_HOST_MEM
        CUDA_CHECK(cudaMallocHost(&data, nbytes));
#else
        data = malloc(nbytes);
#endif
        CHECK(data) << "\nMalloc mem: " << nbytes << " bytes failed.";
        return data;
    }

    /*! \brief Zero-Reset the memory */
    static void Memset(
        size_t              nbytes,
        void*               ptr) {
        memset(ptr, 0, nbytes);
    }

    /*! \brief Zero-Reset the memory asynchronously */
    void MemsetAsync(
        size_t              nbytes,
        void*               ptr) {
        memset(ptr, 0, nbytes);
    }

    /*! \brief Copy the memory */
    template<class DstContext, class SrcContext>
    static void Memcpy(
        size_t              nbytes,
        void*               dst,
        const void*         src) {
        memcpy(dst, src, nbytes);
    }

    /*! \brief Copy the memory asynchronously */
    template<class DstContext, class SrcContext>
    void MemcpyAsync(
        size_t              nbytes,
        void*               dst,
        const void*         src) {
        memcpy(dst, src, nbytes);
    }

    /*! \brief Copy the memory with given type asynchronously */
    template<typename T, class DstContext, class SrcContext>
    void Copy(
        int                 n,
        T*                  dst,
        const T*            src) {
        if (dst == src) return;
        if (std::is_fundamental<T>::value)
            Memcpy<DstContext, SrcContext>(
                n * sizeof(T), (void*)dst, (const void*)src);
        else for (int i = 0; i < n; i++) dst[i] = src[i];
    }

    /*! \brief Free the memory */
    static void Delete(void* data) { free(data); }

    /*! \brief Return the device id */
    int device_id() const { return 0; }

    /*! \brief Return the stream id */
    int stream_id() const { return 0; }

    /*! \brief Set the stream id */
    void set_stream_id(int stream_id) {}

    /*! \brief Return the internal random generator */
    std::mt19937* rand_generator() {
        if (!rand_generator_.get())
            rand_generator_.reset(new std::mt19937(random_seed_));
        return rand_generator_.get();
    }

 private:
    /*! \brief Store the random seed */
    unsigned int random_seed_;

    /*! \brief Store the internal random generator */
    unique_ptr<std::mt19937> rand_generator_;
};

#define CPU_FP16_NOT_SUPPORTED \
    LOG(FATAL) << "FP16 is unsupported for CPUContext.";

}  // namepsace dragon

#endif  // DRAGON_CORE_CONTEXT_H_