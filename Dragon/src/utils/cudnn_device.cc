#ifdef WITH_CUDNN

#include "core/types.h"
#include "core/tensor.h"
#include "utils/cudnn_device.h"

namespace dragon {

float CUDNNType<float>::oneval = 1.0;
float CUDNNType<float>::zeroval = 0.0;
const void* CUDNNType<float>::one =
static_cast<void *>(&CUDNNType<float>::oneval);
const void* CUDNNType<float>::zero =
static_cast<void *>(&CUDNNType<float>::zeroval);

double CUDNNType<double>::oneval = 1.0;
double CUDNNType<double>::zeroval = 0.0;
const void* CUDNNType<double>::one =
static_cast<void *>(&CUDNNType<double>::oneval);
const void* CUDNNType<double>::zero =
static_cast<void *>(&CUDNNType<double>::zeroval);

#ifdef WITH_CUDA_FP16

float CUDNNType<float16>::oneval = 1.0;
float CUDNNType<float16>::zeroval = 0.0;
const void* CUDNNType<float16>::one =
static_cast<void*>(&CUDNNType<float16>::oneval);
const void* CUDNNType<float16>::zero =
static_cast<void*>(&CUDNNType<float16>::zeroval);

#endif

template <typename T>
void cudnnSetTensorDesc(cudnnTensorDescriptor_t* desc, const vector<TIndex>& dims) {
    int ndim = (int)dims.size();
    int* dimA = new int[ndim];
    int* strideA = new int[ndim];
    TIndex stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strideA[i] = stride;
        dimA[i] = dims[i];
        stride *= dimA[i];
    }
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc, CUDNNType<T>::type, ndim, dimA, strideA));
    delete[] dimA;
    delete[] strideA;
}

template <typename T>
void cudnnSetTensorDesc(cudnnTensorDescriptor_t* desc, 
    const vector<TIndex>& dims, const vector<TIndex>& strides) {
    CHECK_EQ(dims.size(), strides.size());
    CHECK(dims.size() >= 3 && dims.size() <= 8);
    int ndim = (int)dims.size();
    int* dimA = new int[ndim];
    int* strideA = new int[ndim];
    for (int i = ndim - 1; i >= 0; i--) {
        strideA[i] = strides[i];
        dimA[i] = dims[i];
    }
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc, CUDNNType<T>::type, ndim, dimA, strideA));
    delete[] dimA;
    delete[] strideA;
}

template <typename T>
void cudnnSetTensorDesc(cudnnTensorDescriptor_t* desc, Tensor* tensor) {
    //  cuDNN requires ndim from 3 to 8
    //  we fake a reshaped dims to pass check
    vector<TIndex> fake_dims(tensor->dims());
    if (fake_dims.size() < 3 || fake_dims.size() > 8) {
        fake_dims.assign({ 1, 1 });
        fake_dims.push_back(tensor->count());
    }
    cudnnSetTensorDesc<T>(desc, fake_dims);
}

template void cudnnSetTensorDesc<float>(cudnnTensorDescriptor_t*, Tensor*);
template void cudnnSetTensorDesc<float>(cudnnTensorDescriptor_t*, const vector<TIndex>&);
template void cudnnSetTensorDesc<float>(cudnnTensorDescriptor_t*, const vector<TIndex>&, const vector<TIndex>&);


template void cudnnSetTensorDesc<double>(cudnnTensorDescriptor_t*, Tensor*);
template void cudnnSetTensorDesc<double>(cudnnTensorDescriptor_t*, const vector<TIndex>&);
template void cudnnSetTensorDesc<double>(cudnnTensorDescriptor_t*, const vector<TIndex>&, const vector<TIndex>&);


#ifdef WITH_CUDA_FP16
template void cudnnSetTensorDesc<float16>(cudnnTensorDescriptor_t*, Tensor*);
template void cudnnSetTensorDesc<float16>(cudnnTensorDescriptor_t*, const vector<TIndex>&);
template void cudnnSetTensorDesc<float16>(cudnnTensorDescriptor_t*, const vector<TIndex>&, const vector<TIndex>&);
#endif

}    // namespace dragon

#endif    // WITH_CUDNN
