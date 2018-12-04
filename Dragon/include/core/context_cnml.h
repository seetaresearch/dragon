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

#ifndef DRAGON_CORE_CONTEXT_CNML_H_
#define DRAGON_CORE_CONTEXT_CNML_H_

/*! CAMBRICON's CNRT && CNML Environment */

#include "core/common.h"

struct cnrtStream;
struct cnmlCpuTensor;
struct cnmlTensor;
struct cnmlFusionOp;
typedef struct cnrtStream* cnrtStream_t;
typedef struct cnmlCpuTensor* cnmlCpuTensor_t;
typedef struct cnmlTensor* cnmlTensor_t;
typedef struct cnmlFusionOp* cnmlFusionOp_t;

namespace dragon {

class CNRTObject;

class CNMLContext {
 public:
     CNMLContext(const DeviceOption& option)
        : device_id_(option.device_id()),
        random_seed_(option.has_random_seed() ?
            option.random_seed() : DEFAULT_RNG_SEED) {
        CHECK_EQ(option.device_type(), CNML);
    }

    CNMLContext(const int device_id = 0)
        : device_id_(device_id),
          random_seed_(DEFAULT_RNG_SEED) {}

    void SwitchToDevice(int stream_id);

    inline void SwitchToDevice() { SwitchToDevice(1); }

    void FinishDeviceCompution();

    static void* New(size_t nbytes);

    static void Memset(
        size_t              nbytes,
        void*               ptr);

    inline void MemsetAsync(
        size_t              nbytes,
        void*               ptr) {
        Memset(nbytes, ptr);
    }

    template<class DstContext, class SrcContext>
    static void Memcpy(
        size_t              nbytes,
        void*               dst,
        const void*         src);

    template<class DstContext, class SrcContext>
    inline void MemcpyAsync(
        size_t              nbytes,
        void*               dst,
        const void*         src) {
        Memcpy<DstContext, SrcContext>(dst, src, nbytes);
    }

    static void Delete(void* data);

    inline int device_id() const { return device_id_; }

    inline void set_stream_id(int stream_id) { stream_id_ = stream_id; }

    inline cnrtStream_t cnrt_stream() {
        return cnrt_stream(device_id_, stream_id_);
    }

    static cnrtStream_t cnrt_stream(
        int                 device_id,
        int                 stream_id);

    static std::mutex& mutex() { static std::mutex m; return m; }

    static CNRTObject* cuda_object();

 private:
    int device_id_, stream_id_ = 1, random_seed_;
    unique_ptr<std::mt19937> rand_generator_;
};

}  // namepsace dragon

#endif  // DRAGON_CORE_CONTEXT_CNML_H_