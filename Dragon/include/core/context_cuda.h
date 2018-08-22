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

#ifndef DRAGON_CORE_CONTEXT_CUDA_H_
#define DRAGON_CORE_CONTEXT_CUDA_H_

#include "core/common.h"
#include "core/context.h"
#include "utils/cuda_device.h"
#include "utils/cudnn_device.h"

namespace dragon {

#ifdef WITH_CUDA

class CUDAObject {
 public:
    CUDAObject() {
        for (int i = 0; i < CUDA_MAX_DEVICES; i++) {
            cuda_streams[i] = vector<cudaStream_t>();
            cublas_handles[i] = vector<cublasHandle_t>();
#ifdef WITH_CUDNN
            cudnn_handles[i] = vector<cudnnHandle_t>();
#endif
        }
    }

    ~CUDAObject() {
        for (int i = 0; i < CUDA_MAX_DEVICES; i++) {
            for (int j = 0; j < cuda_streams[i].size(); j++) {
                auto& stream = cuda_streams[i][j];
                //  follow the caffe2, do not check the stream destroying
                //  Error code 29 (driver shutting down) is inevitable
                //  TODO(PhyscalX): Can someone solve this issue?
                if (stream) cudaStreamDestroy(stream);
            }
            for (auto& handle : cublas_handles[i])
                if (handle) { CUBLAS_CHECK(cublasDestroy_v2(handle)); }
#ifdef WITH_CUDNN
            for (auto& handle : cudnn_handles[i])
                if (handle) { CUDNN_CHECK(cudnnDestroy(handle)); }
#endif
        }
    }

    //  follow the caffe2,
    //  each device takes a group of non-bl0cking streams
    //  the stream 0 is reserved for default stream,
    //  as some computations really require it,
    //  e.g. cublas.asum() and mixed cpu/cuda operations
    //  besides, somes calls, such as cudnn.conv() and cudnn.rnn(),
    //  produce wrong results if running them on non-blocking streams
    //  note that caffe2 also use default streams (within CuDNNState)
    cudaStream_t GetStream(int device_id, int stream_id) {
        vector<cudaStream_t>& dev_streams = cuda_streams[device_id];
        if (dev_streams.size() <= (unsigned)stream_id)
            dev_streams.resize(stream_id + 1, nullptr);
        if (!dev_streams[stream_id]) {
            DeviceGuard guard(device_id);
            unsigned int flags = !stream_id ?
                cudaStreamDefault : cudaStreamNonBlocking;
            CUDA_CHECK(cudaStreamCreateWithFlags(
                &dev_streams[stream_id], flags));
        } return dev_streams[stream_id];
    }

    cublasHandle_t GetCuBLASHandle(int device_id, int stream_id) {
        vector<cublasHandle_t>& dev_handles = cublas_handles[device_id];
        if (dev_handles.size() <= (unsigned)stream_id)
            dev_handles.resize(stream_id + 1, nullptr);
        if (!dev_handles[stream_id]) {
            DeviceGuard guard(device_id);
            CUBLAS_CHECK(cublasCreate_v2(&dev_handles[stream_id]));
            CUBLAS_CHECK(cublasSetStream_v2(dev_handles[stream_id],
                GetStream(device_id, stream_id)));
#if CUDA_VERSION >= 9000
            if (TENSOR_CORE_AVAILABLE())
                CUBLAS_CHECK(cublasSetMathMode(
                    dev_handles[stream_id], CUBLAS_TENSOR_OP_MATH));
#endif
        } return dev_handles[stream_id];
    }

#ifdef WITH_CUDNN
    cudnnHandle_t GetCuDNNHandle(int device_id, int stream_id) {
        vector<cudnnHandle_t>& dev_handles = cudnn_handles[device_id];
        if (dev_handles.size() <= (unsigned)stream_id)
            dev_handles.resize(stream_id + 1, nullptr);
        if (!dev_handles[stream_id]) {
            DeviceGuard guard(device_id);
            CUDNN_CHECK(cudnnCreate(&dev_handles[stream_id]));
            CUDNN_CHECK(cudnnSetStream(dev_handles[stream_id],
                GetStream(device_id, stream_id)));
        } return dev_handles[stream_id];
    }
#endif

    vector<cudaStream_t> cuda_streams[CUDA_MAX_DEVICES];
    vector<cublasHandle_t> cublas_handles[CUDA_MAX_DEVICES];
#ifdef WITH_CUDNN
    vector<cudnnHandle_t> cudnn_handles[CUDA_MAX_DEVICES];
#endif
};

class CUDAContext {
 public:
    CUDAContext(const DeviceOption& option)
        : device_id_(option.device_id()),
          random_seed_(option.has_random_seed() ?
              option.random_seed() : DEFAULT_RNG_SEED) {
        CHECK_EQ(option.device_type(), CUDA);
    }

    CUDAContext(const int device_id = 0)
        : device_id_(device_id),
          random_seed_(DEFAULT_RNG_SEED) {}

    inline void SwitchToDevice(int stream_id) {
        CUDA_CHECK(cudaSetDevice(device_id_));
        stream_id_ = stream_id;
    }

    inline void SwitchToDevice() { SwitchToDevice(1); }

    inline void FinishDeviceCompution() {
        cudaStreamSynchronize(cuda_stream());
        cudaError_t error = cudaGetLastError();
        CHECK_EQ(error, cudaSuccess)
            << "\nCUDA Error: " << cudaGetErrorString(error);
    }

    inline static void* New(size_t nbytes) {
        void* data;
        cudaMalloc(&data, nbytes);
        CHECK(data) << "Malloc cuda mem: " 
                    << nbytes << " bytes failed.";
        return data;
    }

    inline static void Memset(
        size_t              nbytes,
        void*               ptr) {
        CUDA_CHECK(cudaMemset(ptr, 0, nbytes));
    }

    inline void MemsetAsync(
        size_t              nbytes,
        void*               ptr) {
        CUDA_CHECK(cudaMemsetAsync(ptr, 0,
            nbytes, cuda_stream()));
    }

    template<class DstContext, class SrcContext>
    inline static void Memcpy(
        size_t              nbytes,
        void*               dst,
        const void*         src) {
        CUDA_CHECK(cudaMemcpy(dst, src, nbytes,
            cudaMemcpyDefault));
    }

    template<class DstContext, class SrcContext>
    inline void MemcpyAsync(
        size_t              nbytes,
        void*               dst,
        const void*         src) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes,
            cudaMemcpyDefault, cuda_stream()));
    }

    template<typename T, class DstContext, class SrcContext>
    inline void Copy(
        int                 n,
        T*                  dst,
        const T*            src) {
        if (dst == src) return;
        MemcpyAsync<SrcContext, DstContext>(
            n * sizeof(T), (void*)dst, (const void*)src);
    }

    inline static void Delete(void* data) { cudaFree(data); }

    inline int device_id() const { return device_id_; }

    inline void set_stream_id(int stream_id) { stream_id_ = stream_id; }

    inline cudaStream_t cuda_stream() {
        return cuda_stream(device_id_, stream_id_);
    }

    static cudaStream_t cuda_stream(
        int                 device_id,
        int                 stream_id) {
        return cuda_object_.GetStream(device_id, stream_id);
    }

    cublasHandle_t cublas_handle() {
        return cuda_object_.GetCuBLASHandle(device_id_, stream_id_);
    }

    inline std::mt19937* rand_generator() {
        if (!rand_generator_.get())
            rand_generator_.reset(new std::mt19937(random_seed_));
        return rand_generator_.get();
    }

    curandGenerator_t& curand_generator() {
        if (!curand_generator_) {
            DeviceGuard guard(device_id_);
            CURAND_CHECK(curandCreateGenerator(
                &curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
                curand_generator_, random_seed_));
        }
        CURAND_CHECK(curandSetStream(
            curand_generator_, cuda_stream()));
        return curand_generator_;
    }

#ifdef WITH_CUDNN
    cudnnHandle_t cudnn_handle() {
        return cuda_object_.GetCuDNNHandle(device_id_, stream_id_);
    }
#endif

    static std::mutex& mutex() { static std::mutex m; return m; }

    static thread_local CUDAObject cuda_object_;

 private:
    int device_id_, stream_id_ = 1, random_seed_;
    unique_ptr<std::mt19937> rand_generator_;
    curandGenerator_t curand_generator_ = nullptr;
};

template <class Context>
class CUDAClosure {
 public:
    CUDAClosure() {}
    explicit CUDAClosure(Context* ctx): ctx_(ctx) {}

    void Sync() {
        for (auto stream_id : active_streams_) {
            cudaStreamSynchronize(cuda_object_
                .GetStream(ctx_->device_id(), stream_id));
            cudaError_t error = cudaGetLastError();
            CHECK_EQ(error, cudaSuccess)
                << "\nCUDA Error: " << cudaGetErrorString(error);
        }
        active_streams_.clear();
    }

    inline cudaStream_t cuda_stream(int stream_id) {
        active_streams_.push_back(stream_id);
        return cuda_object_.GetStream(
            ctx_->device_id(), stream_id);
    }

    inline cublasHandle_t cublas_handle(int stream_id) {
        active_streams_.push_back(stream_id);
        return cuda_object_.GetCuBLASHandle(
            ctx_->device_id(), stream_id);
    }

#ifdef WITH_CUDNN
    inline cudnnHandle_t cudnn_handle(int stream_id) {
        active_streams_.push_back(stream_id);
        return cuda_object_.GetCuDNNHandle(
            ctx_->device_id(), stream_id);
    }
#endif

 protected:
    Context* ctx_;
    CUDAObject cuda_object_;
    vector<int> active_streams_;
};

#else  // WITH_CUDA

class CUDAContext {
 public:
    CUDAContext(const DeviceOption& option) { CUDA_NOT_COMPILED; }
    CUDAContext(const int device_id = 0) { CUDA_NOT_COMPILED; }

    inline void SwitchToDevice() { CUDA_NOT_COMPILED; }
    inline void SwitchToDevice(int stream_id) { CUDA_NOT_COMPILED; }

    inline void FinishDeviceCompution() { CUDA_NOT_COMPILED; }

    inline static void Memset(
        size_t              nbytes,
        void*               ptr) {
        CUDA_NOT_COMPILED;
    }

    inline void MemsetAsync(
        size_t              nbytes,
        void*               ptr) {
        CUDA_NOT_COMPILED;
    }

    template<class DstContext, class SrcContext>
    inline static void Memcpy(
        size_t              nbytes,
        void*               dst,
        const void*         src) {
        CUDA_NOT_COMPILED;
    }

    template<class DstContext, class SrcContext>
    inline void MemcpyAsync(
        size_t              nbytes,
        void*               dst,
        const void*         src) {
        CUDA_NOT_COMPILED;
    }

    inline int device_id() const { return 0; }
    inline void set_stream_id(int stream_id) {}
};

#endif // WITH_CUDA

}    // namespace dragon

#endif    // DRAGON_CORE_CONTEXT_CUDA_H_